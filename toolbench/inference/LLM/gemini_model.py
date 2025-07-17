#!/usr/bin/env python
# coding=utf-8
from typing import Optional, List, Mapping, Any
from termcolor import colored
import json
import random
import requests
from typing import Optional
from toolbench.model.model_adapter import get_conversation_template
from toolbench.utils import process_system_message
from toolbench.inference.utils import SimpleChatIO, react_parser
from toolbench.inference.Prompts.ReAct_prompts import FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION_ZEROSHOT


class Gemini:
    def __init__(self, model="gemini-2.5-flash", gemini_key="", temperature=0.5, top_p=1) -> None:
        super().__init__()
        self.model = model
        self.gemini_key = gemini_key
        self.endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.gemini_key}"
        )
        self.temperature = temperature
        self.top_p = top_p

    def prediction(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        max_try = 10
        stop_seqs = stop

        while True:
            try:
                print(stop)
                body = {
                    "contents": [
                        { "parts": [ { "text": prompt } ] }
                    ],
                    "generationConfig": {
                        "stopSequences":    stop_seqs,
                        "temperature":      self.temperature,
                        "topP":             self.top_p,
                        "frequencyPenalty": 0.0,
                        "presencePenalty":  0.0
                    }
                }
                resp = requests.post(self.endpoint, json=body, timeout=15)
                j = resp.json()
                result = j["candidates"][0]["content"]["parts"][0]["text"]
                # usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

                print("──> Gemini prompt:\n", prompt)
                print("──> Gemini response:\n", result)
                break
            except Exception as e:
                print(e)
                max_try -= 1
                if max_try < 0:
                    result = "Exceed max retry times. Please check your davinci api calling."
                    break

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
    llm = Gemini()
    result = llm.prediction("How old are you?")
    print(result)