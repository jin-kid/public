from preprocessing import feature_exploration


def reduce():
    # feature_exploration engineering
    feature_exploration(is_train_data=True, change_down_size=True, down_size=2000)


if __name__ == '__main__':
    reduce()
