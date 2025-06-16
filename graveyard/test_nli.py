from transformers import pipeline

pipe = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device="mps",
    hypothesis_template="These APIs are {}."
)

text = (
    "API A: Formula 1 Standings: F1 Constructor and Drivers Standings. | Driver Standings: Will return the current F1 season driver standings.\n"
    "API B: FIA Formula 1 Championship Statistics: FIA Formula 1 Championship Statistics is a REST API. | Drivers Standings: Use this endpoint to retrieve drivers standings data about a specifit F1 championship by specifying a year. If you ommit the ***year*** query parameter, a default value will be set to current year. The response data will contain information about the position in the rank list, driver's name, nationality, team and total points accumulated."
)

result = pipe(
    text,
    candidate_labels=["equivalent", "not equivalent"]
)

print(result)