#!/usr/bin/env python
# coding=utf-8
from typing import Optional, List, Mapping, Any, Tuple
from termcolor import colored
import json
import random
import openai
from toolbench.model.model_adapter import get_conversation_template
from toolbench.utils import process_system_message
from toolbench.inference.utils import SimpleChatIO, react_parser
from toolbench.inference.Prompts.ReAct_prompts import FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION_ZEROSHOT

from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import time

class Claude:
    def __init__(self, model: str = "claude-v1", anthropic_api_key: str = "", temperature=0.5, top_p=1) -> None:
        super().__init__()
        self.client = Anthropic(api_key=anthropic_api_key)
        self.model = model
        self.chatio = SimpleChatIO()
        self.temperature = temperature
        self.top_p = top_p

    def prediction(self, prompt: str, stop: Optional[List[str]] = None
                  ) -> Tuple[str, int]:
        """
        Wrap the user-prompt in Claude's HUMAN/AI delimiters,
        ask for up to 512 tokens, stop on the next human turn.
        """

        max_retries = 5
        backoff = 1.0

        for attempt in range(1, max_retries + 1):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    messages=[
                        {"role": "user",   "content": prompt}
                    ],
                    max_tokens=512,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    stop_sequences=[HUMAN_PROMPT],          # halt when Claude would expect a human turn
                )
                break
            except:
                if attempt == max_retries:
                    raise
                print(f"[Claude] Overloaded, retry {attempt}/{max_retries} after {backoff}s")
                time.sleep(backoff)
                backoff *= 2
        else:
            # Should never hit
            raise RuntimeError("Claude prediction failed after retries")

        print("──> Claude prompt:\n", prompt)
        text = resp.content[0].text.strip()
        print("──> Claude response:\n", text)

        # usage estimates
        # prompt_toks     = getattr(resp.usage, "input_tokens", 0)
        # completion_toks = getattr(resp.usage, "output_tokens", 0)
        # total_toks      = prompt_toks + completion_toks

        # usage = {
        #     "prompt_tokens": prompt_toks,
        #     "completion_tokens": completion_toks,
        #     "total_tokens": total_toks
        # }

        return text

    def add_message(self, message):
        self.conversation_history.append(message)

    def change_messages(self,messages):
        self.conversation_history = messages

    def display_conversation(self, detailed=False):
        role_to_color = {
            "system": "red",
            "user": "green",
            "assistant": "blue",
            "function": "magenta",
        }
        print("before_print"+"*"*50)
        for message in self.conversation_history:
            print_obj = f"{message['role']}: {message['content']} "
            if "function_call" in message.keys():
                print_obj = print_obj + f"function_call: {message['function_call']}"
            print_obj += ""
            print(
                colored(
                    print_obj,
                    role_to_color[message["role"]],
                )
            )
        print("end_print"+"*"*50)

    def parse(self,functions,process_id,**args):
        template = "tool-llama-single-round"
        conv = get_conversation_template(template)
        if template == "tool-llama":
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        elif template == "tool-llama-single-round" or template == "tool-llama-multi-rounds":
            roles = {"system": conv.roles[0], "user": conv.roles[1], "function": conv.roles[2], "assistant": conv.roles[3]}

        conversation_history = self.conversation_history
        prompt = ''
        for message in conversation_history:
            role = roles[message['role']]
            content = message['content']
            if role == "System" and functions != []:
                content = process_system_message(content, functions)
            prompt += f"{role}: {content}\n"
        prompt += "Assistant:\n"
        
        if functions != []:
            predictions = self.prediction(prompt)
        else:
            predictions = self.prediction(prompt)

        function_names = [fn["name"] for fn in functions]

        # react format prediction
        thought, action, action_input = react_parser(predictions, function_names)
        message = {
            "role": "assistant",
            "content": thought,
            "function_call": {
                "name": action,
                "arguments": action_input
            }
        }
        return message, 0, 0


if __name__ == "__main__":
    claude = Claude()
    claude.change_messages([
      {"role": "system",    "content": ""},
      {"role": "user",      "content": "What time is it in Tokyo?"}
    ])
    msg, code, tokens = claude.parse(functions=[], process_id=0)
    print(msg, code, tokens)