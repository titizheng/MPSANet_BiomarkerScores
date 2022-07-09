import sys
import os
# import argparse
# import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

from data.annotation import Formatter  # noqa
'''XML files of tumor tissue regions annotated by pathologists using ASAP and other software were converted into JSON'''


def run(xml_path, json_path):
    Formatter.camelyon16xml2json(xml_path, json_path)


def main(xml_path, json_path):

    run(xml_path, json_path)


if __name__ == '__main__':

    xml_path_root = "/home/omnisky/ZTT_file/data/CD30ICH/NCRF-data/json/test/xmlcontrast"
    json_path_root = "/home/omnisky/ZTT_file/data/CD30ICH/NCRF-data/json/test/contrast"
    for xml_name in os.listdir(xml_path_root):
        xml_path = os.path.join(xml_path_root, xml_name)
        print("xml的路径：{}".format(xml_path))
        (name, extension) = os.path.splitext(xml_name)
        json_path = os.path.join(json_path_root, name + ".json")
        print("json的路径：{}".format(json_path))
        main(xml_path, json_path)

