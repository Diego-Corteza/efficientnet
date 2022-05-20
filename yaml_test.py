import yaml


def yaml_loader(filepath):
    """
    loads a yaml file in given path
    :param filepath:
    :return: data
    """
    with open(filepath, "r") as file_descriptor:
        data = yaml.load(file_descriptor)
    return data


def yaml_dump(filepath, data):
    """
    Dumps data to a yaml file
    :param filepath:
    :param data:
    :return:
    """
    with open(filepath, "w") as file_descriptor:
        yaml.dump(data, file_descriptor)


if __name__ == "__main__":
    filepath = "settings.yaml"
    data = yaml_loader(filepath)
    print(data)
