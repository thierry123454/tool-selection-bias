from transformers import AutoTokenizer
import json

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)

prompt = '''System: You are AutoGPT, you can use many tools(functions) to do the following task.
First I will give you the task description, and your task start.
At each step, you need to give your thought to analyze the status now and what to do next, with a function call to actually excute your step. Your output should follow this format:
Thought:
Action
Action Input:

After the call, you will get the call result, and you are now in a new state.
Then you will analyze your status now, then decide what to do next...
After many (Thought-call) pairs, you finally perform the task, then you can give your finial answer.
Remember:
1.the state change is irreversible, you can't go back to one of the former state, if you want to restart the task, say "I give up and restart".
2.All the thought is short, at most in 5 sentence.
3.You can do more then one trys, so if your plan is to continusly try some conditions, you can do one of the conditions per try.
Let's Begin!
Task description: You should use functions to help handle the real time user querys. Remember:
1.ALWAYS call "Finish" function at the end of the task. And the final answer should contain enough information to show to the user,If you can't handle the task, or you find that function calls always fail(the function is not valid now), use function Finish->give_up_and_restart.
2.Do not use origin tool names, use only subfunctions' names.
You have access of the following tools:
1.text_language_by_api_ninjas: Detect the language from any input text. See more info at https://api-ninjas.com/api/textlanguage.
2.translate_v3: Easy and reliable Machine Translation  and Language Detection
3.translate_all_languages: Translate All Language -  Text Translator100x cheaper than Google Translate. Same API. Same quality.  Translate All Languages provides a simple API for translating plain text between any of 100+ supported languages. If you don’t know what language the text is written in, our API will detect the language of the original request.  telegram DM: @justapi1
4.what_s_language: Detect the language of a given text
5.quick_language_detector: Feed this API a few sentences and have it determine what language it is with a confidence score.

Specifically, you have access to the following APIs: [{'name': 'v1_textlanguage_for_text_language_by_api_ninjas', 'description': 'This is the subfunction for tool "text_language_by_api_ninjas", you can use this tool.The description of this function is: "API Ninjas Text Language API endpoint"', 'parameters': {'type': 'object', 'properties': {'text': {'type': 'string', 'description': '', 'example_value': 'hello world!'}}, 'required': ['text'], 'optional': []}}, {'name': 'fast_language_detection_for_translate_v3', 'description': 'This is the subfunction for tool "translate_v3", you can use this tool.The description of this function is: "This endpoint will return the Language of the Text"', 'parameters': {'type': 'object', 'properties': {'text': {'type': 'string', 'description': '', 'example_value': "this is accurate and it can improve if it's longer"}}, 'required': ['text'], 'optional': []}}, {'name': 'detect_for_translate_all_languages', 'description': 'This is the subfunction for tool "translate_all_languages", you can use this tool.The description of this function is: "detect_for_translate_all_languagess the language of text within a request."', 'parameters': {'type': 'object', 'properties': {'text': {'type': 'string', 'description': 'The input text upon which to perform language detection. Repeat this parameter to perform language detection on multiple text inputs.', 'example_value': 'If you don’t know what language the text is written in, our API will detect the language of the original request.'}}, 'required': ['text'], 'optional': []}}, {'name': 'languagedetection_for_what_s_language', 'description': 'This is the subfunction for tool "what_s_language", you can use this tool.The description of this function is: "Detect the language of a given text and return the detected language code"', 'parameters': {'type': 'object', 'properties': {'text': {'type': 'string', 'description': '', 'example_value': 'How to Identify the Language of any Text'}}, 'required': ['text'], 'optional': []}}, {'name': 'detect_language_for_quick_language_detector', 'description': 'This is the subfunction for tool "quick_language_detector", you can use this tool.The description of this function is: "Feed this API a few sentences and have it determine what language it is with a confidence score"', 'parameters': {'type': 'object', 'properties': {'text': {'type': 'string', 'description': '', 'example_value': "Cela peut identifier 52 langues humaines à partir d'échantillons de texte et renvoyer des scores de confiance pour chaque"}, 'detectedcount': {'type': 'integer', 'description': '', 'example_value': '5'}}, 'required': ['text'], 'optional': ['detectedcount']}}, {'name': 'Finish', 'description': 'If you believe that you have obtained a result that can answer the task, please call this function to provide the final answer. Alternatively, if you recognize that you are unable to proceed with the task in the current state, call this function to restart. Remember: you must ALWAYS call this function at the end of your attempt, and the only part that will be shown to the user is the final answer, so it should contain sufficient information.', 'parameters': {'type': 'object', 'properties': {'return_type': {'type': 'string', 'enum': ['give_answer', 'give_up_and_restart']}, 'final_answer': {'type': 'string', 'description': 'The final answer you want to give the user. You should have this field if "return_type"=="give_answer"'}}, 'required': ['return_type']}}]
User:
Can you detect the language of this text: "Bonjour, comment allez-vous aujourd'hui?"?
Begin!

Assistant:
'''

ids = tok(prompt, add_special_tokens=False).input_ids

print(len(ids))