import sys
import os
# import argparse
# import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
''' First:Convert the positive area xml file obtained by the annotation software into a json format file'''

from KFwsi.data.annotation import Formatter  # noqa



def run(xml_path, json_path):
    Formatter.camelyon16xml2json(xml_path, json_path)

def main(xml_path, json_path):
    # logging.basicConfig(level=logging.INFO)

    #
    run(xml_path, json_path)


if __name__ == '__main__':


    xml_path_root =r".\Annotation"
    json_path_root =r".\jsons"
    # xml_name='61691(CD30).xml'
    for xml_name in os.listdir(xml_path_root):
        xml_path = os.path.join(xml_path_root, xml_name)
        print("xml path：{}".format(xml_path))
        (name, extension) = os.path.splitext(xml_name)
        json_path = os.path.join(json_path_root, name + ".json")
        print("json path：{}".format(json_path))
        main(xml_path, json_path)

