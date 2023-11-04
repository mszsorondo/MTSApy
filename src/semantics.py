from transformers import AutoConfig, AutoModel, AutoTokenizer
from torch import flatten, stack
model = AutoModel.from_pretrained("bert-base-cased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", padding='max_length', truncation=True)

numbers = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten"]


tokens_by_number = [tokenizer(f"{num} out of ten philosophers ate", return_tensors="pt") for num in numbers]
embeds_by_number = stack([flatten(model(**token).last_hidden_state) for token in tokens_by_number])
breakpoint()


inputs_cero_diez = tokenizer("Zero out of ten philosophers ate", return_tensors="pt")
inputs_cinco_diez = tokenizer("Five out of ten philosophers ate", return_tensors="pt")
inputs_diez_diex = tokenizer("Ten out of ten philosophers ate",  return_tensors="pt")

outputs_cero_diez = model(**inputs_cero_diez).last_hidden_state
outputs_cinco_diez = model(**inputs_cinco_diez).last_hidden_state
outputs_diez_diez = model(**inputs_cero_diez).last_hidden_state


flatten(outputs_cinco_diez).dot(flatten(outputs_diez_diez).T)
breakpoint()


context = ("FSP stands for 'Finite State Process' and it is a language that is used to model discrete event systems."
           "A system is modelled as a set of machines, each one of them with their own states, state changes are triggered"
           " by events. Events can be controllable or uncontrollable. At each point in discrete time, a machine can only be on one state."
           "The whole system's state is given by the set of all of the states from the system's machines. An event triggered by one "
           " machine can trigger a state change in another machine (or multiple) in the state. This happens if the triggered event is also possible"
           "for the other machine given its current state. "
           "At every time-step, an event happens. This system can be modelled as an attributed graph where nodes are system states "
           "and edges are events. System and machine states are also labeled as 'marking' or as 'non-marking'. A system state is marking"
           "if and only if all of its machine states are marking."
           "Here is an FSP script that models a problem parameterized by N and K, K models the complexity of each isolated machine and N"
           "regulates the number of machines. Note that the state complexity is O(K^N). For compiling this file, one should fix N and K"
           " within the file before using them. For example: const N = 3 \nconst K = 6.")

with open("data/prompt_engineering/DirectorForNonBlocking_benchmark_generic_code/AirTrafficControl.lts", 'r') as f:
    fsp_code = f.read()

compostate_info = ""
#FIXME text is way too long
inputs = tokenizer(context+fsp_code+compostate_info, return_tensors="pt")

outputs = model(**inputs)


breakpoint()
pass
#model = AutoModel.from_pretrained("bert-base-cased", output_attentions=True)