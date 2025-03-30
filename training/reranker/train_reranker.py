import argparse
import torch
from datasets import load_dataset
from sentence_transformers import (
    CrossEncoder,
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
)
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--pos_weight', type=float)
    parser.add_argument('--label_mapping', choices=['s', 'p'])
    parser.add_argument('--max_seq_length', type=int)
    parser.add_argument('--fp16', action='store_true')
    args = parser.parse_args()

    dataset = load_dataset("json", data_files=args.dataset_name)

    def map_labels(ex):
        target_label = 'similar_sentence' if args.label_mapping == "s" else 'paraphrase'
        label = 1 if ex['label'] == target_label else 0
        return {
            'text1': ex['sentence1'],
            'text2': ex['sentence2'],
            'label': label
        }

    train_samples = dataset['train'].map(map_labels)
    train_samples = train_samples.remove_columns(
        [col for col in train_samples.column_names if col not in {'text1', 'text2', 'label'}]
    )
    from collections import Counter
    print(Counter(train_samples['label']))
    pos_weight_tensor = torch.tensor([args.pos_weight]) if args.pos_weight else None

    model = CrossEncoder(args.model_name, num_labels=1, max_length=args.max_seq_length)
    loss_fn = BinaryCrossEntropyLoss(model=model, pos_weight=pos_weight_tensor)

    training_args = CrossEncoderTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        eval_strategy='no',
        save_strategy='no',
        logging_strategy='epoch',
        logging_dir='logs',
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        dataloader_drop_last=True,
        dataloader_num_workers=4,
    )

    trainer = CrossEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_samples,
        tokenizer=model.tokenizer,
        loss=loss_fn,
    )

    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    main()

