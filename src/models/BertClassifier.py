# code based on
# https://iq.opengenus.org/binary-text-classification-bert/
# https://colab.research.google.com/drive/1Wd8pQDaSwLgyHsHF9UjN7-fP93cfPOFI?usp=sharing

import os
import shutil
import numpy as np
import pandas as pd
import re
import torch
import random
import time
import datetime

from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig, \
    get_linear_schedule_with_warmup
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

_root_path = '../../'


class BertDataset(object):
    """
    Class for Training and Validation Set
    """

    def __init__(self, sentences, labels, batch_size=50, max_len=200):
        # low level BERT
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        def encode_sent(sent):
            # Encode sentence and adding special tokens '[CLS]' and '[SEP]'
            return tokenizer.encode(sent, add_special_tokens=True)

        inputs = list(map(encode_sent, sentences))

        print('Max sentence length:', max([len(sen) for sen in inputs]))
        print('Pad sentences to length:', max_len)

        inputs = pad_sequences(inputs, maxlen=max_len, truncating="post", padding="post")

        attention_masks = list(map(lambda s: [int(token_id > 0) for token_id in s], inputs))

        # changing the numpy arrays into tensors
        inputs = torch.tensor(inputs)
        labels = torch.tensor(labels)
        masks = torch.tensor(attention_masks)

        # DataLoader
        self.data = TensorDataset(inputs, masks, labels)
        self.sampler = RandomSampler(self.data)
        self.dataloader = DataLoader(self.data, sampler=self.sampler, batch_size=batch_size)


class BertClassifier(object):
    def __init__(self, model='bert-base-uncased', num_classes=2, max_len=200):

        # init model
        if os.path.isfile(model):
            # load from file
            self.model = BertForSequenceClassification.from_pretrained(model)
        else:
            # download pretrained
            self.model = BertForSequenceClassification.from_pretrained(
                model,
                num_labels=num_classes,
                output_attentions=False,
                output_hidden_states=False,
            )
        # init tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model, do_lower_case=True)
        self.max_len = max_len

    def summary(self):
        # Get all of the model's parameters as a list of tuples.
        params = list(self.model.named_parameters())
        print('The BERT model has {:} different named parameters.\n'.format(len(params)))
        print('==== Embedding Layer ====\n')
        for p in params[0:5]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
        print('\n==== First Transformer ====\n')
        for p in params[5:21]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
        print('\n==== Output Layer ====\n')
        for p in params[-4:]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    def do_train(self, epochs, train_dataloader, validation_dataloader=None, optimizer=None, save_path=None):
        """
        Method for training BertClassifier
        """
        # init optimizer
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = AdamW(self.model.parameters(), lr=2e-5, eps=1e-8)

        # select device (CPU or GPU)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print('Using GPU:', torch.cuda.get_device_name(0))
        else:
            print('No GPU available, using the CPU instead.')
            self.device = torch.device("cpu")
        self.model.to(self.device)

        total_steps = len(train_dataloader) * epochs
        global_step = 0

        scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                    num_warmup_steps=0,  # Default value in run_glue.py
                                                    num_training_steps=total_steps)

        # This training code is based on the `run_glue.py` script here:
        # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

        # Set the seed value all over the place to make this reproducible.
        set_seed(42)
        # Store the average loss after each epoch so we can plot them.
        loss_values = []

        for epoch_i in range(epochs):
            # Perform one full pass over the training set.
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))

            # Measure how long the training epoch takes.
            t0 = time.time()

            # Put the model into training mode
            self.model.train()

            # For each batch of training data...
            for step, batch in enumerate(train_dataloader):

                # Add batch to GPU
                batch = tuple(t.to(self.device) for t in batch)

                # Unpack the inputs from our dataloader
                b_input_ids, b_input_mask, b_labels = batch

                #  clear previously calculated gradients before performing a backward pass
                self.model.zero_grad()

                # Perform a forward pass
                outputs = self.model(b_input_ids,
                                     attention_mask=b_input_mask,
                                     token_type_ids=None,
                                     labels=b_labels)
                loss = outputs[0]

                # Save training loss
                loss_values.append(loss.item())

                # Perform a backward pass to calculate the gradients.
                loss.backward()

                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Update parameters and take a step using the computed gradient.
                self.optimizer.step()

                # Update the learning rate.
                scheduler.step()

                global_step += 1
                # Progress update every 100 batches.
                if step % 100 == 0 and not step == 0:
                    # Report progress.
                    print(f'Batch: {step:>5,} / {len(train_dataloader):>5,} - '
                          f'Moving Average Loss: {round(sum(loss_values[-100:])/100, 4)} - '
                          f'Elapsed: {format_time(time.time() - t0)} - '
                          f'ETA: {format_time((time.time() - t0) / global_step * (total_steps - step))}')

            # Calculate the average loss over the training data.
            avg_train_loss = sum(loss_values) / len(train_dataloader)

            print("")
            print(f"Average training loss: {avg_train_loss:.2f}")
            print(f"Training epoch took: {format_time(time.time() - t0)}")

            if save_path is not None:
                os.makedirs(save_path, exist_ok=True)
                self.model.save_pretrained(f'{save_path}model_{global_step}')
                self.tokenizer.save_pretrained(f'{save_path}model_{global_step}')

            if validation_dataloader is not None:
                self.do_evaluation(validation_dataloader)

        print("")
        print("Training complete!")

    def do_evaluation(self, validation_dataloader):
        print("")
        print("Running Validation")

        t0 = time.time()

        # Put the model in evaluation mode
        self.model.eval()

        # Tracking variables
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(self.device) for t in batch)

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients, saving memory and
            # speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                outputs = self.model(b_input_ids,
                                     token_type_ids=None,
                                     attention_mask=b_input_mask)
            logits = outputs[0]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences.
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)

            # Accumulate the total accuracy.
            eval_accuracy += tmp_eval_accuracy

            # Track the number of batches
            nb_eval_steps += 1

        # Report the final accuracy for this validation run.
        print(f"Accuracy: {eval_accuracy / nb_eval_steps:.2f}")
        print(f"Validation took: {format_time(time.time() - t0)}")

    def predict(self, input):
        if type(input) == str:
            input = list(input)

        inputs = list(map(lambda s: self.tokenizer.encode(s, add_special_tokens=True), input))
        inputs = pad_sequences(inputs, maxlen=self.max_len, truncating="post", padding="post")
        masks = list(map(lambda s: [int(token_id > 0) for token_id in s], inputs))

        inputs = torch.tensor(inputs)
        masks = torch.tensor(masks)

        output = self.model(inputs,
                            attention_mask=masks)

        output = torch.nn.Softmax()(output[0]).detach().numpy()

        output = [dict(zip(['neg', 'pos'], p)) for p in output]

        if len(output) == 1:
            return output[0]

        return output

    def save(self, path):
        self.model.save_pretrained(path)

    def load(self, path):
        self.__init__(path)


def flat_accuracy(preds, labels):
    """
    Function to calculate the accuracy of our predictions vs labels
    """
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def set_seed(seed):
    """
    Set a seed to be reproducible
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
