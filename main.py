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


def prepare_data(data_lists, size):
    prepared_data = []
    for data_list in data_lists:
        random.shuffle(data_list)
        prepared_data += data_list[:size]

    random.shuffle(prepared_data)
    return prepared_data


def calculate_accuracy(net, data_list):
    n_attempts = 0
    n_correct = 0

    for label, x in data_list:
        y = net(x)
        guess = torch.distributions.Categorical(y).sample()

        n_attempts += 1
        if guess.item() == label[0].item():
            n_correct += 1

    return float(n_correct) / float(n_attempts)


def main():
    # ---------- HYPERPARAMETERS ---------- #

    N_EPISODES = 150
    BATCH_SIZE = 30

    LR = 0.005
    HIDDEN_SIZE = 500

    # ------------------------------------- #

    DEVICE = torch.device("cuda")

    dm = data_manager.DataManager()

    charlist = dm.get_charlist()

    data = []
    for i, o in enumerate(dm.get_origins()):
        label = torch.tensor([i], dtype=torch.long).to(DEVICE)
        data.append(
            [(label, word_to_tensor(w, charlist).to(DEVICE)) for w in dm.get_names(o)]
        )

    validation_size = 10
    validation_data = []
    for i in range(len(data)):
        random.shuffle(data[i])
        validation_data += data[i][:validation_size]
        data[i] = data[i][validation_size:]

    datasize = min([len(entry) for entry in data])

    # ------------------------------------- #

    net = aitools.nn.rnn.NetworkLSTM(
        len(charlist), HIDDEN_SIZE, len(data), torch.nn.Softmax(dim=0), device=DEVICE
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
                for label, x in batch:
                    try:
                        y = net(x).view((1, -1))
                    except RuntimeError:
                        print("err")
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
