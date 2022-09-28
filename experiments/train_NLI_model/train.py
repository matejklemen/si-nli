import json
import logging
import os
import sys
from argparse import ArgumentParser

import pandas as pd
import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from transformers import AutoTokenizer

from slo_nli.data.data_loader import SloNLITransformersDataset
from slo_nli.models.nli import TransformersNLITrainer

import numpy as np
import matplotlib.pyplot as plt

parser = ArgumentParser()
parser.add_argument("--experiment_dir", type=str, default="debug")
parser.add_argument("--pretrained_name_or_path", type=str, default="EMBEDDIA/sloberta")

parser.add_argument("--train_path", type=str, help="Path to the training set")
parser.add_argument("--dev_path", type=str, help="Path to the validation set")
parser.add_argument("--test_path", type=str, help="Path to the test set")

parser.add_argument("--num_epochs", type=int, default=3)
parser.add_argument("--max_seq_len", type=int, default=84)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--early_stopping_rounds", type=int, default=5)
parser.add_argument("--validate_every_n_examples", type=int, default=100)

parser.add_argument("--use_cpu", action="store_true")


if __name__ == "__main__":
    args = parser.parse_args()
    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)

    with open(os.path.join(args.experiment_dir, "experiment_config.json"), "w") as f:
        json.dump(vars(args), fp=f, indent=4)

    RANDOM_SEED = 17
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # Set up logging to file and stdout
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for curr_handler in [logging.StreamHandler(sys.stdout),
                         logging.FileHandler(os.path.join(args.experiment_dir, "experiment.log"))]:
        curr_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s"))
        logger.addHandler(curr_handler)

    for k, v in vars(args).items():
        v_str = str(v)
        v_str = f"...{v_str[-(50 - 3):]}" if len(v_str) > 50 else v_str
        logging.info(f"|{k:30s}|{v_str:50s}|")

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_name_or_path)
    tokenizer.save_pretrained(args.experiment_dir)

    train_set = SloNLITransformersDataset(args.train_path, tokenizer=tokenizer,
                                          max_length=args.max_seq_len, return_tensors="pt")
    dev_set = SloNLITransformersDataset(args.dev_path, tokenizer=tokenizer,
                                        max_length=args.max_seq_len, return_tensors="pt")
    test_set = SloNLITransformersDataset(args.test_path, tokenizer=tokenizer,
                                         max_length=args.max_seq_len, return_tensors="pt")

    logging.info(f"Loaded {len(train_set)} training examples, "
                 f"{len(dev_set)} dev examples and "
                 f"{len(test_set) if test_set is not None else 0} test examples")

    trainer = TransformersNLITrainer(args.experiment_dir,
                                     pretrained_model_name_or_path=args.pretrained_name_or_path,
                                     num_labels=len(train_set.label_names),
                                     batch_size=args.batch_size,
                                     learning_rate=args.learning_rate,
                                     validate_every_n_steps=args.validate_every_n_examples,
                                     early_stopping_tol=args.early_stopping_rounds,
                                     optimized_metric="accuracy",
                                     device=("cuda" if not args.use_cpu else "cpu"))

    trainer.run(train_dataset=train_set, val_dataset=dev_set, num_epochs=args.num_epochs)

    if test_set is not None:
        trainer = TransformersNLITrainer.from_pretrained(args.experiment_dir)
        test_res = trainer.evaluate(test_set)
        if hasattr(test_set, "labels"):
            np_labels = test_set.labels.numpy()
            np_pred = test_res["pred_label"].numpy()
            np_pred_proba = test_res["pred_proba"].numpy()

            # Save predictions to file
            pred_data = {"pred_label": np_pred}
            for _lbl, _idx in test_set.label2idx.items():
                pred_data[f"proba_{_lbl}"] = np_pred_proba[:, _idx]
            pd.DataFrame(pred_data).to_csv(os.path.join(args.experiment_dir, "predictions.tsv"), index=False, sep="\t")

            conf_matrix = confusion_matrix(y_true=np_labels, y_pred=np_pred)
            plt.matshow(conf_matrix, cmap="Blues")
            for (i, j), v in np.ndenumerate(conf_matrix):
                plt.text(j, i, v, ha='center', va='center',
                         bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
            plt.xticks(np.arange(len(test_set.label_names)), test_set.label_names)
            plt.yticks(np.arange(len(test_set.label_names)), test_set.label_names)
            plt.xlabel("(y_pred)")

            plt.savefig(os.path.join(args.experiment_dir, "confusion_matrix.png"))
            logging.info(f"Confusion matrix:\n {conf_matrix}")

            model_metrics = {
                "accuracy": accuracy_score(y_true=np_labels, y_pred=np_pred),
                "macro_precision": precision_score(y_true=np_labels, y_pred=np_pred, average="macro"),
                "macro_recall": recall_score(y_true=np_labels, y_pred=np_pred, average="macro"),
                "macro_f1": f1_score(y_true=np_labels, y_pred=np_pred, average="macro")
            }

            for curr_pos_label, idx_curr_pos in test_set.label2idx.items():
                true_bin_labels = (np_labels == idx_curr_pos).astype(np.int32)
                pred_bin_labels = (np_pred == idx_curr_pos).astype(np.int32)

                model_metrics[f"precision_{curr_pos_label}"] = precision_score(y_true=true_bin_labels,
                                                                               y_pred=pred_bin_labels,
                                                                               average="binary")
                model_metrics[f"recall_{curr_pos_label}"] = recall_score(y_true=true_bin_labels,
                                                                         y_pred=pred_bin_labels,
                                                                         average="binary")
                model_metrics[f"f1_{curr_pos_label}"] = f1_score(y_true=true_bin_labels,
                                                                 y_pred=pred_bin_labels,
                                                                 average="binary")

                conf_matrix = confusion_matrix(y_true=true_bin_labels, y_pred=pred_bin_labels)
                plt.matshow(conf_matrix, cmap="Blues")
                for (i, j), v in np.ndenumerate(conf_matrix):
                    plt.text(j, i, v, ha='center', va='center',
                             bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
                plt.xticks([0, 1], [f"not_{curr_pos_label}", curr_pos_label])
                plt.yticks([0, 1], [f"not_{curr_pos_label}", curr_pos_label])
                plt.xlabel("(y_pred)")

                plt.savefig(os.path.join(args.experiment_dir, f"bin_confusion_matrix_{curr_pos_label}.png"))
                logging.info(f"Confusion matrix (positive={curr_pos_label}):\n {conf_matrix}")

            with open(os.path.join(args.experiment_dir, "metrics.json"), "w") as f_metrics:
                logging.info(model_metrics)
                json.dump(model_metrics, fp=f_metrics, indent=4)

            logging.info(model_metrics)
        else:
            logging.info(f"Skipping test set evaluation because no labels were found!")
