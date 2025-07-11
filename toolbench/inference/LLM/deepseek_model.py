#!/usr/bin/env python
# coding=utf-8
from typing import Optional, List, Mapping, Any
from termcolor import colored
import json
import random
from openai import OpenAI
from typing import Optional
from toolbench.model.model_adapter import get_conversation_template
from toolbench.utils import process_system_message
from toolbench.inference.utils import SimpleChatIO, react_parser
from toolbench.inference.Prompts.ReAct_prompts import FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION_ZEROSHOT


class DeepSeek:
    def __init__(self, model="deepseek-chat", deepseek_key="") -> None:
        super().__init__()
        self.model = model
        self.openai_key = deepseek_key
        self.client = OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com")

    def prediction(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        max_try = 10
        while True:
            try:
                print("──> DeepSeek prompt:\n", prompt)
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role":"system", "content": ""},
                        {"role":"user",   "content": prompt},
                    ],
                    temperature=0.5,
                    max_tokens=512,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=["End Action"],
                )
                result = response.choices[0].message.content.strip()
                print("──> DeepSeek response:\n", result)
                break
            except Exception as e:
                print(e)
                max_try -= 1
                if max_try < 0:
                    result = "Exceed max retry times. Please check your davinci api calling."
                    break
        # usage = {
        #     "prompt_tokens":     response.usage.prompt_tokens,
        #     "completion_tokens": response.usage.completion_tokens,
        #     "total_tokens":      response.usage.total_tokens,
        # }
        return result
        
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
    llm = DeepSeek()
    result = llm.prediction("How old are you?")
    print(result)