import datasets
glue_dataset = datasets.load_dataset('glue', 'mrpc')
print(len(glue_dataset['train']))