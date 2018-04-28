import csv
import matplotlib.pyplot as plt

# Given a column of record and the invalid number of N/A, compute the mean by remove some maximum and some minimum
def compute_mean(each_column, invalid_num):
    # filter the data, get rid of N/A
    valid_data = []
    for i in range(len(each_column)):
        try:
            epoch = int(each_column[i])
            valid_data.append(epoch)
        except ValueError:
            continue
    valid_data = sorted(valid_data)
    # print(valid_data)
    valid_data = valid_data[int(invalid_num):len(valid_data)]
    valid_data = valid_data[0:(len(valid_data)-int(invalid_num))]
    # print(valid_data)
    mean = 0.0
    for i in range(len(valid_data)):
        mean += valid_data[i]
    mean = mean / len(valid_data)
    return mean

if __name__ == '__main__':
    filename = "table_data.txt"
    with open(filename, "r") as f:
        reader = csv.reader(f, delimiter='\t')
        total_rows = len(open(filename).readlines())
        # use first row to get the column number
        first_row = next(reader)
        num_cols = len(first_row)
        dataset = []
        for i in range(num_cols):
            dataset.append([])

        # -2, because the real data is not at the first and last row
        for i in range(total_rows - 2):
            each_row = next(reader)
            for k in range(num_cols):
                dataset[k].append(each_row[k])

        lastrow = next(reader)
        invalid_count = []
        for i in range(num_cols):
            invalid_count.append(lastrow[i])

        labels = first_row[1:len(first_row)]
        index = []
        for i in range(len(labels)):
            index.append(float(labels[i].split('=')[1]))

        means = []
        for i in range(1, num_cols):
            # index.append(i)
            means.append(compute_mean(dataset[i], invalid_count[i]))

        print(means)
        print(index)
        # print(means.reverse())
        fig, ax1 = plt.subplots()
        ax1.set_title("average number of epochs needed to reach popErr = 0.05 for iris dataset")
        ax1.plot(index, means)
        ax1.set_xlabel('learning rate')
        ax1.set_ylabel('average number of epochs')
        plt.savefig("average_epochs.png")