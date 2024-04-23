import json
filename = './data/all_courses_v2.json'
# Read the JSON data from the file
with open(filename, 'r') as file:
    data = json.load(file)

# Extract the 'text' values into a new list
docs_processed = [item['text'] for item in data]
print(len(docs_processed))