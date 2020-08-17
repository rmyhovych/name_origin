import torch
import random
import matplotlib.pyplot as plt

import aitools
import data_manager


def word_to_tensor(word: str, chars: list):
    t = torch.zeros(size=(len(word), len(chars)))
    for i, c in enumerate(word):
        t[i][chars.index(c)] = 1.0

    return t


def tensor_to_word(t, chars: list):
    w = []
    for i in t:
        m = torch.argmax(i)
        w.append(chars[int(m.item())])

    return "".join(w)


def prepare_data(data_list, size):
    prepared_data = []
    for i, entry in enumerate(data_list):
        random.shuffle(entry[1])
        prepared_data += [
            (word_tensor, torch.tensor([i], dtype=torch.long))
            for word_tensor in entry[1][:size]
        ]

    random.shuffle(prepared_data)
    return prepared_data


def calculate_accuracy(net, data_list):
    n_attempts = 0
    n_correct = 0

    for i, entry in enumerate(data_list):
        for x in entry[1]:
            y = net(x)
            guess = torch.distributions.Categorical(y).sample()

            n_attempts += 1
            if guess.item() == i:
                n_correct += 1
    return float(n_correct) / float(n_attempts)


def main():
    # ---------- HYPERPARAMETERS ---------- #

    N_EPISODES = 150
    BATCH_SIZE = 30

    LR = 0.005
    HIDDEN_SIZE = 100

    # ------------------------------------- #

    dm = data_manager.DataManager()

    charlist = dm.get_charlist()
    data = [
        [o, [word_to_tensor(w, charlist) for w in dm.get_names(o)]]
        for o in dm.get_origins()
    ]

    validation_size = 10

    validation_data = []
    for data_entry in data:
        random.shuffle(data_entry[1])
        validation_data.append([data_entry[0], data_entry[1][:validation_size]])
        data_entry[1] = data_entry[1][validation_size:]

    datasize = min([len(entry[1]) for entry in data])

    # ------------------------------------- #

    net = aitools.nn.rnn.NetworkLSTM(
        len(charlist), HIDDEN_SIZE, len(data), torch.nn.Softmax(dim=0)
    )
    optim = torch.optim.Adam(net.parameters(), lr=LR)
    lossF = torch.nn.functional.cross_entropy

    losses = []
    accuracies = []
    try:
        for episode in range(N_EPISODES):
            avrg_loss = 0.0
            n_batches = 0

            training_data = prepare_data(data, datasize)
            for batch_start in range(0, len(training_data), BATCH_SIZE):
                batch = training_data[batch_start : batch_start + BATCH_SIZE]
                n_batches += 1

                cumulative_loss = 0.0
                for x, label in batch:
                    y = net(x).view((1, -1))
                    cumulative_loss += lossF(y, label)

                optim.zero_grad()
                avrg_loss += cumulative_loss.item() / len(batch)
                cumulative_loss.backward()
                optim.step()

            avrg_loss /= n_batches
            losses.append(avrg_loss)
            accuracy = calculate_accuracy(net, validation_data)
            accuracies.append(accuracy)

            print("{}\t: {} - {}".format(episode, avrg_loss, accuracy))
    except KeyboardInterrupt:
        pass

    plt.plot(losses)
    plt.plot(accuracies)
    plt.show()


if __name__ == "__main__":
    main()
