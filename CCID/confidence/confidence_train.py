import warnings

from .models.loss_function import SSE
from .models.ConvNet_region import ConvNet_region
from .models.DnCNN import DnCNN

import torch
from torch.serialization import SourceChangeWarning
from torch.utils.data import DataLoader

import csv
import numpy as np
import os
import sys

from CCID.confidence.data_generation import (
    ImagesDataset, testing_dataset_path, training_dataset_path, saved_testing_data_path, saved_training_data_path)

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 50  # change me
batch_size = 100
learning_rate = 0.001
weight_decay = 1e-4
num_w = 8
test_model = len(sys.argv) > 1 and "true" in sys.argv[1].lower()
debug = True


def main():
    # Ignore source code change warnings
    warnings.filterwarnings("ignore", category=SourceChangeWarning)

    torch.manual_seed(42)  # Reproducibility

    model = ConvNet_region(n_channels=3).to(device)

    # Loss and optimizer
    criterion = SSE(1, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_set = ImagesDataset(dataset_path=training_dataset_path,
                              saved_dataset_dir=saved_training_data_path,
                              aug_count=2,
                              noise_count=15,
                              verbose=debug)
    # train_loader = DataLoader(dataset=train_data, num_workers=4, drop_last=True, batch_size=batch_size, shuffle=True,
    #                           pin_memory=True)

    # test_data = ImagesDataset(testing_dataset_path, saved_testing_data_path, verbose=debug)
    # num_holdout = int(0.3 * len(test_data))

    # test_data_list = torch.utils.data.random_split(test_data, [num_holdout, len(test_data) - num_holdout],
    #                                                generator=torch.Generator().manual_seed(42))
    # test_data = test_data_list[1]
    # holdout_data = test_data_list[0]
    # test_loader = DataLoader(dataset=test_data, shuffle=True, batch_size=batch_size, pin_memory=True)
    # holdout_loader = DataLoader(dataset=holdout_data, shuffle=True, batch_size=batch_size, pin_memory=True)

    num_validation = int(0.3 * len(train_set))
    split_train_list = torch.utils.data.random_split(train_set, [num_validation, len(train_set) - num_validation])

    validation_data = split_train_list[0]
    train_data = split_train_list[1]

    train_loader = DataLoader(dataset=train_data, num_workers=num_w, batch_size=batch_size, shuffle=True,
                              pin_memory=True)

    validation_loader = DataLoader(dataset=validation_data, num_workers=num_w, batch_size=batch_size,
                                   shuffle=True, pin_memory=True)



    def not_exists(*filenames):
        res = True
        for filename in filenames:
            res = res and not os.path.isfile(filename)
        return res

    for i in range(100):
        v_file_name = f"validation_losses_{i}.csv"
        t_file_name = f"train_losses_{i}.csv"
        if not_exists(v_file_name, t_file_name):
            break

    def compute_statistics(set):
        num_samples = 30
        with torch.no_grad(),\
             open("stats_abs.csv", "w", newline='') as abs_csv_file,\
             open("stats_diff.csv", "w", newline='') as diff_csv_file:

            abs_writer = csv.writer(abs_csv_file)
            diff_writer = csv.writer(diff_csv_file)
            total = 0
            print("Loss\t|y-ŷ|\tMedian\t1st\t10th\t90th\t99th\ty-ŷ" +
                  "\tMedian\t1st\t10th\t90th\t99th")
            for images, labels in set:
                labels = labels.to(device)
                images = images.to(device)
                outputs = model(images)
                for i in range(len(images)):
                    label = labels[i]

                    output = outputs[i]
                    total += 1

                    diff = label.sub(output).cpu()

                    abs_diff = torch.abs(diff)

                    loss = criterion(output, label).item()
                    diff_row = [loss]
                    abs_row = [loss]
                    
                    diff_row.extend(diff.view(-1).tolist())
                    abs_row.extend(abs_diff.view(-1).tolist())

                    diff_writer.writerow(diff_row)
                    abs_writer.writerow(abs_row)

                    a, b, c, d, e = percentiles(abs_diff, 50, 1, 10, 90, 99)
                    f, g, h, i, j = percentiles(diff, 50, 1, 10, 90, 99)
                    # print(f"Loss: {loss:2f}\tMedian: {median:2f}\nPercentiles: ")
                    # print(f"1%: {hundredth_percentile:2f}\t10%: {tenth_percentile:2f}")
                    # print(f"90%: {ninetieth_percentile:2f}\t99%: {ninetynineth_percentile:2f}")
                    print(f"{loss:.4f}\t\t{a:.4f}\t{b:.4f}\t{c:.4f}\t{d:.4f}\t{e:.4f}" + \
                          f"\t\t{f:.4f}\t{g:.4f}\t{h:.4f}\t{i:.4f}\t{j:.4f}")
                    if total >= num_samples:
                        return

    def save_individual_losses(set):
        print("Computing individual losses…")
        with open("individual_losses.csv", "w", newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            for i, (images, labels) in enumerate(set):
                if i % 100 == 99:
                    print(f"{i+1}/{len(set)}")
                labels = labels.to(device)
                images = images.to(device)
                outputs = model(images)
                for i in range(len(images)):
                    label = labels[i]
                    output = outputs[i]

                    loss = criterion(output, label).item()
                    csv_writer.writerow([loss])



    def percentiles(data, *ps):
        output = []
        for p in ps:
            output.append(np.percentile(data, p))
        return tuple(output)

    def eval_model_and_report(set, exact=False):
        # Test the model
        num_samples = 10000
        model.eval()  # Evaluation mode
        with torch.no_grad():
            loss = 0
            total = 0
            for i, (images, labels) in enumerate(set):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                total += labels.size(0)
                loss += criterion(outputs, labels).item()
                if exact:
                    if i % 100 == 99:
                        print(f"Computing exact loss… {i + 1}/{len(set)}")
                if not exact and total >= num_samples:
                    break

            if exact:
                print("Average loss of the model on test images: {:.4f}".format(loss / total))
            else:
                print("Estimated loss of the model (over {:d} samples): {:.4f}".format(total, loss / total))

        return loss / total

    if test_model:
        test_set = ImagesDataset(dataset_path=testing_dataset_path,
                                 saved_dataset_dir=saved_testing_data_path,
                                 aug_count=1,
                                 noise_count=5,
                                 verbose=debug)
        test_loader = DataLoader(dataset=test_set, num_workers=num_w, batch_size=batch_size,
                              pin_memory=True)
        # Always use the latest model
        model.load_state_dict(torch.load("model_49.cpkt"))
        compute_statistics(set=test_loader)
        eval_model_and_report(exact=True, set=test_loader)
        save_individual_losses(set=test_loader)
        exit(0)

    print("Starting training…")
    # Train the model
    n = 1
    m = 4
    num_data = len(train_loader)
    with open(v_file_name, 'w') as validation_csv_file, open(t_file_name, 'w') as training_csv_file:
        validation_writer = csv.writer(validation_csv_file)
        test_writer = csv.writer(training_csv_file)
        validation_writer.writerow(["Epoch", "Loss"])
        for epoch in range(num_epochs):
            model.train()  # Training mode
            losses = []
            for i, (images, labels) in enumerate(train_loader):

                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                losses.append(loss.item())
                optimizer.step()

                if (i + 1) % (num_data // m) == 0:
                    avg_loss = sum(losses) / (len(losses) * batch_size)
                    epoch_to_print = 1 + epoch + i / num_data
                    print(f"Epoch: {epoch_to_print:.2f}, Avg. train loss: {avg_loss:.5f}")
                    test_writer.writerow([epoch_to_print, avg_loss])
                    losses = []

            if epoch % n == n - 1:
                print(f"Epoch: {epoch + 1}")
                validation_loss = eval_model_and_report(set=validation_loader)
                validation_writer.writerow([epoch + 1, validation_loss])
                savename = "model_" + str(epoch) + ".cpkt"
                torch.save(model.state_dict(), savename)

    eval_model_and_report(set=validation_loader, exact=True)
    compute_statistics(set=validation_loader)

    # Save the model checkpoint
    torch.save(model.state_dict(), "confidence_map_model.cpkt")


if __name__ == '__main__':
    main()
