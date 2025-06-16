import os
import openai

# make sure you’ve exported your key:
#   export OPENAI_API_KEY="sk-…"
openai.api_key = os.getenv("OPENAI_API_KEY")

def compare_apis(api_a: str, api_b: str, model: str = "gpt-4"):
    """
    Ask the LLM if API A and API B have the same functionality.
    Returns the LLM's raw reply.
    """
    system = (
        "You are an expert at reading API endpoint names and descriptions and "
        "deciding whether two endpoints provide the same functionality. "
        "Answer in JSON with keys 'equivalent' (true/false) and 'explanation'."
    )
    user = (
        f"Here are two API definitions:\n\n"
        f"API A:\n{api_a}\n\n"
        f"API B:\n{api_b}\n\n"
        f"Do these two APIs roughly offer the same functionality?"
    )

    resp = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0,
        max_tokens=150,
    )
    return resp.choices[0].message.content.strip()

if __name__ == "__main__":
    api_a = (
        "Formula 1 Standings: F1 Constructor and Drivers Standings. | Driver Standings: Will return the current F1 season driver standings."
    )
    api_b = (
        "FIA Formula 1 Championship Statistics: FIA Formula 1 Championship Statistics is a REST API. | Drivers Standings: Use this endpoint to retrieve drivers standings data about a specifit F1 championship by specifying a year. If you ommit the ***year*** query parameter, a default value will be set to current year. The response data will contain information about the position in the rank list, driver's name, nationality, team and total points accumulated."
    )

    result = compare_apis(api_a, api_b)
    print(result)