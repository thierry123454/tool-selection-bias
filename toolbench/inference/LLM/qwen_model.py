#!/usr/bin/env python
# coding=utf-8
from typing import Optional, List, Mapping, Any
from termcolor import colored
from openai import OpenAI
from typing import Optional
from toolbench.model.model_adapter import get_conversation_template
from toolbench.utils import process_system_message, process_system_message_debias
from toolbench.inference.utils import SimpleChatIO, react_parser
from toolbench.inference.Prompts.ReAct_prompts import FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION_ZEROSHOT
import json, re

LOG_PATTERN = "v1_textlanguage_for_text_language_by_api_ninjas"
LOG_PATH    = "ninjas_prompts.txt"
LOG_SEP     = "\n<|END_PROMPT|>\n" 
OUT_FILE = "subset_preds.jsonl"

def save_subset(qid: int, resp_text: str):
    try:
        sel = json.loads(resp_text)
        if not isinstance(sel, list):
            raise ValueError
    except Exception:
        m = re.search(r"\[(.*?)\]", resp_text, flags=re.S)
        items = re.findall(r'"([^"]+)"', m.group(0)) if m else []
        sel = [s.strip() for s in items]

    with open(OUT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"query_id": qid, "selected_tools": sel}, ensure_ascii=False) + "\n")
    return sel

class Qwen:
    def __init__(self, model="", qwen_key="", mitigation=False, qid=-1) -> None:
        super().__init__()
        self.model = model
        self.openai_key = qwen_key
        self.log_relevant_prompts = False
        self.client = OpenAI(api_key=qwen_key, base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
        self.mitigation = mitigation
        self.qid = qid

    def _log_if_matches(self, prompt: str):
        if LOG_PATTERN in prompt:
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(prompt)
                f.write(LOG_SEP)

    def prediction(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        max_try = 10

        if self.log_relevant_prompts:
            self._log_if_matches(prompt)
        
        while True:
            try:
                # print(f"──> {self.model} prompt:\n", prompt)
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
                    extra_body={
                        "enable_thinking": False,
                    }
                )
                # response = "YAHOOO!!!!"
                result = response.choices[0].message.content.strip()
                print("──> Qwen response:\n", result)
                break
            except Exception as e:
                print(e)
                max_try -= 1
                if max_try < 0:
                    result = "Exceed max retry times. Please check your api calling."
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
        print(f'Conversation template: {conv}')
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
                if not self.mitigation:
                    content = process_system_message(content, functions)
                else:
                    functions = [f for f in functions if f["name"] != "Finish"]
                    content = process_system_message_debias(content, functions)
            print(f"{role} + {process_id}: {content}\n\n\n\n\n")
            prompt += f"{role}: {content}\n"
        prompt += "Assistant:\n"
        
        predictions = self.prediction(prompt)

        if self.mitigation:
            save_subset(self.qid, predictions)

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
    llm = Qwen()
    result = llm.prediction("How old are you?")
    print(result)