#!/usr/bin/env python
# coding=utf-8
from typing import Optional, List, Mapping, Any, Tuple
from termcolor import colored
import json
import random
import openai
from toolbench.model.model_adapter import get_conversation_template
from toolbench.inference.utils import SimpleChatIO, react_parser
from toolbench.inference.Prompts.ReAct_prompts import FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION_ZEROSHOT

from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import time

class Claude:
    def __init__(self, model: str = "claude-v1", anthropic_api_key: str = "") -> None:
        super().__init__()
        self.client = Anthropic(api_key=anthropic_api_key)
        self.model = model
        self.chatio = SimpleChatIO()

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

        text = resp.content[0].text.strip()

        # usage estimates
        prompt_toks     = getattr(resp.usage, "input_tokens", 0)
        completion_toks = getattr(resp.usage, "output_tokens", 0)
        total_toks      = prompt_toks + completion_toks

        usage = {
            "prompt_tokens": prompt_toks,
            "completion_tokens": completion_toks,
            "total_tokens": total_toks
        }


        return text, usage

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
        conv = get_conversation_template("tool-llama-single-round")
        roles = {"system": conv.roles[0], "user": conv.roles[1], "function": conv.roles[2], "assistant": conv.roles[3]}
        conversation_history = self.conversation_history
        question = ''
        for message in conversation_history:
            role = roles[message['role']]
            content = message['content']
            if role == "User":
                question = content
                break
        func_str = ""
        func_list = []
        for function_dict in functions:
            param_str = ""
            api_name = function_dict["name"]
            func_list.append(api_name)
            if "Finish" in api_name:
                param_str = f'"return_type": string, "final_answer": string, '
                api_desc = "If you believe that you have obtained a result that can answer the task, please call this function to provide the final answer. ALWAYS call this function at the end of your attempt to answer the question finally."
                func_str += f"{api_name}: {api_desc}. Your input should be a json (args json schema): {param_str} The Action to trigger this API should be {api_name} and the input parameters should be a json dict string. Pay attention to the type of parameters.\n\n"
            else:
                api_desc = function_dict["description"][function_dict["description"].find("The description of this function is: ")+len("The description of this function is: "):]
                for param_name in function_dict["parameters"]["properties"]:
                    data_type = function_dict["parameters"]["properties"][param_name]["type"]
                    param_str += f'"{param_name}": {data_type}, '
                param_str = "{{" + param_str + "}}"
                func_str += f"{api_name}: {api_desc}. Your input should be a json (args json schema): {param_str} The Action to trigger this API should be {api_name} and the input parameters should be a json dict string. Pay attention to the type of parameters.\n\n"
        func_list = str(func_list)
        prompt = FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION_ZEROSHOT.replace("{func_str}", func_str).replace("{func_list}", func_list).replace("{func_list}", func_list).replace("{question}", question)
        prompt = prompt.replace("{{", "{").replace("}}", "}")
        for message in conversation_history:
            role = roles[message['role']]
            content = message['content']
            if role == "Assistant":
                prompt += f"\n{content}\n"
            elif role == "Function":
                prompt += f"Observation: {content}\n"
        if functions != []:
            predictions, usage = self.prediction(prompt)
        else:
            predictions, usage = self.prediction(prompt)
        
        # react format prediction
        thought, action, action_input = react_parser(predictions)
        message = {
            "role": "assistant",
            "content": thought,
            "function_call": {
                "name": action,
                "arguments": action_input
            }
        }
        return message, 0, usage["total_tokens"]


if __name__ == "__main__":
    claude = Claude()
    claude.change_messages([
      {"role": "system",    "content": ""},
      {"role": "user",      "content": "What time is it in Tokyo?"}
    ])
    msg, code, tokens = claude.parse(functions=[], process_id=0)
    print(msg, code, tokens)