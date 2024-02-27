'''

This program aims to evaluate the fairness of GPT-2, an older version of GPT-4 (commonly used in ChatGPT and other similar large language models today).
We will evaluate it on three metrics:
- Toxicity, as quantified by a pretrained hate speech classification model
- Regard, an estimation of language polarity towards a particular demographic group
- HONEST score, derived from a dataset in the original 2021 paper (DOI: 10.18653/v1/2021.naacl-main.191) to measure hurtful text continuations

'''

# Required libraries: Huggingface Transformers, Evaluate, Datasets
import random
import evaluate
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer
from random import sample
from collections import Counter

def get_random_prompts(dataset, num_examples=100):
    # Function: Return a random subset of a dataset
    # Args: (dataset: Huggingface Dataset containing prompts) (num_examples: number of elements to retrieve, default to 100)
    # Return: subset of elements in the dataset: list of length num_examples
    
    num_examples = min(max(1, num_examples), len(dataset))
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    return dataset.shuffle()[0:num_examples]

def evaluate_toxicity(text_generation, tokenizer): # Prompts: Real Toxicity Prompts stratified on 4 levels of toxic degeneration
    # Function: Evaluate the toxicity level of a dataset
    # Args: (text_generation: the LLM used to generate continuations) (tokenizer: helper function for segmenting and processing text)
    
    toxicity_prompts = load_dataset("allenai/real-toxicity-prompts", split="train")
    toxic_sample= get_random_prompts(toxicity_prompts)
    toxic_prompts = [p['text'] for p in toxic_sample['prompt']]

    model_continuations=[]
    for prompt in toxic_prompts:
        generation = text_generation(prompt, max_length=50, do_sample=False, pad_token_id=50256)
        continuation = generation[0]['generated_text'].replace(prompt,'')
        model_continuations.append(continuation)

    print('Generated '+ str(len(model_continuations))+ ' continuations')

    toxicity = evaluate.load("toxicity")

    toxicity_ratio = toxicity.compute(predictions=model_continuations, aggregation="ratio")

    toxicity_comp = toxicity.compute(predictions=model_continuations)["toxicity"]
    toxicity_comp = sorted(zip(model_continuations, toxicity_comp), key=lambda x: x[1])

def evaluate_regard(text_generation, tokenizer): # Prompts: BOLD dataset evaluating fairness in open-end languages
    # Function: Evaluate the regard of a dataset
    # Args: (text_generation: the LLM used to generate continuations) (tokenizer: helper function for segmenting and processing text)

    bold = load_dataset("AlexaAI/bold", split="train")
    from random import sample
    female_bold = (sample([p for p in bold if p['category'] == 'American_actresses'],50))
    male_bold = (sample([p for p in bold if p['category'] == 'American_actors'],50))
    male_prompts = [p['prompts'][0] for p in male_bold]
    female_prompts = [p['prompts'][0] for p in female_bold]

    male_continuations=[]
    for prompt in male_prompts:
        generation = text_generation(prompt, max_length=50, do_sample=False, pad_token_id=50256)
        continuation = generation[0]['generated_text'].replace(prompt,'')
        male_continuations.append(continuation)

    print('Generated '+ str(len(male_continuations))+ ' male continuations')

    female_continuations=[]
    for prompt in female_prompts:
        generation = text_generation(prompt, max_length=50, do_sample=False, pad_token_id=50256)
        continuation = generation[0]['generated_text'].replace(prompt,'')
        female_continuations.append(continuation)

    print('Generated '+ str(len(female_continuations))+ ' female continuations')

    regard = evaluate.load('regard', 'compare')
    regard.compute(data = male_continuations, references= female_continuations)
    regard.compute(data = male_continuations, references= female_continuations, aggregation = 'average')

    regard = evaluate.load('regard')
    cats = ["race", "profession", "political_ideology", "religious_ideology"]
    for str in cats:
        print_avgs(str, text_generation, tokenizer, regard, bold)


def print_avgs(d, text_generation, tokenizer, regard, bold):
  # Function: Output the evaluated regard level for particular demographic groups
  # Args: (d: string of demographic group category) (text_generation: the LLM used to generate continuations) (tokenizer: helper function for segmenting and processing text) (regard: model to evaluate regard scoring of a dataset) (bold: dataset to prompt evaluate model)
  
  domain = []

  for i in bold:
    if(i['domain'] == d):
      domain.append(i)

  domain_dict = {}

  for r in domain:
      category = r['category']
      if category in domain_dict.keys():
        domain_dict[category].append(r)
      else:
        domain_dict[category] = [r]


  for r in domain_dict.keys():
    r_list = domain_dict[r]
    domain_dict[r] = sample([p for p in r_list if p['category'] == r], min(20, len(r_list)))
    # print(len(race_dict[r]))
    domain_dict[r] = [p['prompts'][0] for p in domain_dict[r]]
    continuations = []

    for prompt in domain_dict[r]:
      generation = text_generation(prompt, max_length=50, do_sample=False, pad_token_id=50256)
      continuation = generation[0]['generated_text'].replace(prompt,'')
      continuations.append(continuation)

    print(r)
    print(sorted(regard.compute(data = continuations, aggregation = 'average')['average_regard'].items()))
    print()

def evaluate_honest(text_generation, tokenizer): # Prompts: HONEST dataset from aforementioned paper
    # Function: Evaluate the HONEST score of a dataset
    # Args: (text_generation: the LLM used to generate continuations) (tokenizer: helper function for segmenting and processing text)
    honest_dataset = load_dataset("MilaNLProc/honest", 'en_queer_nonqueer', split='honest')
    categories= [h['category'] for h in honest_dataset]
    Counter(categories)
    queer_prompts= sample([h['template_masked'].replace(' [M].','') for h in honest_dataset if h['category'].startswith('queer')], 50)
    nonqueer_prompts= sample([h['template_masked'].replace(' [M].','') for h in honest_dataset if h['category'].startswith('nonqueer')], 50)
    
    size = 20

    queer_continuations = generate_continuations(queer_prompts, size, text_generation, tokenizer)
    nonqueer_continuations = generate_continuations(nonqueer_prompts, size, text_generation, tokenizer)

    honest = evaluate.load('honest', 'en')

    groups = ['queer'] * 50 * size + ['nonqueer'] * 50 * size # expand since we do k sequences for each prompt
    continuations = [c.split() for c in queer_continuations] + [q.split() for q in nonqueer_continuations]

    honest_score = honest.compute(predictions=continuations, groups = groups)
    print(honest_score)

def generate_continuations(honest_prompts: list, size, text_generation, tokenizer):
  # Function: Generate a list of continuations for each element in a list of prompts
  # Args: (honest_prompts: list of prompts from HONEST dataset) (size: number of prompts in list) (text_generation: the LLM used to generate continuations) (tokenizer: helper function for segmenting and processing text)
  # Return: the list of continuations corresponding to the input prompts
  
  continuations = []

  for prompt in honest_prompts:
    generation = text_generation(prompt, max_length=len(tokenizer(prompt))+size, do_sample=True, pad_token_id=50256, num_return_sequences=size)
    for i in range(size):
      continuation = generation[i]['generated_text'].replace(prompt,'')
      continuations.append(continuation)

  return continuations

def main():
    text_generation = pipeline("text-generation", model="gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    evaluate_toxicity(text_generation, tokenizer)
    evaluate_regard(text_generation, tokenizer)
    evaluate_honest(text_generation, tokenizer)

