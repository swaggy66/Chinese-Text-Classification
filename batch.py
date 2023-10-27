import argparse
import json
import os
import platform
import shutil
from copy import deepcopy

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from transformers.trainer_utils import set_seed
from peft import AutoPeftModelForCausalLM

DEFAULT_CKPT_PATH = '/data/wuchengyan/Firefly-master/Qwen-7B-Chat'
#DEFAULT_CKPT_PATH = '/data/wuchengyan/Qwen-main/Qwen-14B-Chat'
_WELCOME_MSG = '''\
Welcome to use Qwen-Chat model, type text to start chat, type :h to show command help.
(欢迎使用 Qwen-Chat 模型，输入内容即可进行对话，:h 显示命令帮助。)

Note: This demo is governed by the original license of Qwen.
We strongly advise users not to knowingly generate or allow others to knowingly generate harmful content, including hate speech, violence, pornography, deception, etc.
(注：本演示受Qwen的许可协议限制。我们强烈建议，用户不应传播及不应允许他人传播以下内容，包括但不限于仇恨言论、暴力、色情、欺诈相关的有害信息。)
'''
_HELP_MSG = '''\
Commands:
    :help / :h          Show this help message              显示帮助信息
    :exit / :quit / :q  Exit the demo                       退出Demo
    :clear / :cl        Clear screen                        清屏
    :clear-his / :clh   Clear history                       清除对话历史
    :history / :his     Show history                        显示对话历史
    :seed               Show current random seed            显示当前随机种子
    :seed <N>           Set random seed to <N>              设置随机种子
    :conf               Show current generation config      显示生成配置
    :conf <key>=<value> Change generation config            修改生成配置
    :reset-conf         Reset generation config             重置生成配置
'''

def _load_model_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path, trust_remote_code=True, resume_download=True,
    )

    if args.cpu_only:
        device_map = "cpu"
    else:
        device_map = "auto"

    model = AutoPeftModelForCausalLM.from_pretrained(
        "/data/wuchengyan/Qwen-main/output_qwenonestage7b1016",
        device_map="auto",
        trust_remote_code=True
    ).eval()

    config = GenerationConfig.from_pretrained(
        args.checkpoint_path, trust_remote_code=True, resume_download=True,
    )

    return model, tokenizer, config

def _gc():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _clear_screen():
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")

def _print_history(history):
    terminal_width = shutil.get_terminal_size()[0]
    print(f'History ({len(history)})'.center(terminal_width, '='))
    for index, (query, response) in enumerate(history):
        print(f'User[{index}]: {query}')
        print(f'QWen[{index}]: {response}')
    print('=' * terminal_width)

def main():
    parser = argparse.ArgumentParser(
        description='QWen-Chat command-line interactive chat demo.')
    parser.add_argument("-c", "--checkpoint-path", type=str, default=DEFAULT_CKPT_PATH,
                        help="Checkpoint name or path, default to %(default)r")
    parser.add_argument("-s", "--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--cpu-only", action="store_true", help="Run demo with CPU only")
    parser.add_argument("--input-file", type=str, help="Input JSON file")
    parser.add_argument("--output-file", type=str, help="Output JSON file")
    args = parser.parse_args()

    history = []

    model, tokenizer, config = _load_model_tokenizer(args)
    orig_gen_config = deepcopy(model.generation_config)

    _clear_screen()
    print(_WELCOME_MSG)

    seed = args.seed

    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            conversations = json.load(f)

        for conv in conversations:
            user_input = conv["conversations"][0]["value"]
            set_seed(seed)
            try:
                for response in model.chat_stream(tokenizer, user_input, history=history, generation_config=config):
                    history.append((user_input, response))
                    # Print the current conversation
                    _clear_screen()
                    print(f"\nUser: {user_input}")
                    print(f"\nQwen-Chat: {response}")
                    break  # Assuming you only need the first response
            except KeyboardInterrupt:
                print('[WARNING] Generation interrupted')
                continue

        if args.output_file:
            output_data = {"conversation": []}
            for user_input, response in history:
                output_data["conversation"].append({"human": user_input, "assistant": response})

            with open(args.output_file, "w", encoding="utf-8") as output_file:
                json.dump(output_data, output_file, ensure_ascii=False, indent=2)

    print('\nAll conversations processed. Type :q to exit.')

    while True:
        command = input().strip()
        if command in [':q', ':exit', ':quit']:
            break

if __name__ == "__main__":
    main()
