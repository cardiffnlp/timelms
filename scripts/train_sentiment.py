import numpy as np
import evaluate

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import load_dataset, concatenate_datasets


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, average='macro')


dataset = load_dataset('tweet_eval', 'sentiment')


MODEL_NAME = 'cardiffnlp/twitter-roberta-base-sep2022'  # change to desired model from the hub
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# augment train set with test set, for downstream apps only - DO NOT EVALUATE ON TEST
# tokenized_datasets['train+test'] = concatenate_datasets([tokenized_datasets['train'],
#                                                          tokenized_datasets['test']])

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

training_args = TrainingArguments(
    do_eval=True,
    evaluation_strategy='epoch',
    output_dir='test_trainer',
    logging_dir='test_trainer',
    logging_strategy='epoch',
    save_strategy='epoch',
    num_train_epochs=10,
    learning_rate=1e-05,
    per_gpu_train_batch_size=16,
    per_gpu_eval_batch_size=16,
    load_best_model_at_end=True,
    metric_for_best_model='recall',
)

metric = evaluate.load('recall')  # default metric for sentiment dataset is recall (macro)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.create_model_card()
trainer.save_model('saved_model')

# res = trainer.evaluate(tokenized_datasets['test'])
# print(res)
