import xml.etree.ElementTree as ET 
def read_xml():
    tree=ET.parse('/home/makalo/workspace/code/kaggle/Faster-RCNN_TF/data/demo/1.xml')
    objs=tree.findall('object')
    coor=[]
    for i,obj in enumerate(objs):
        bbox=obj.find('bndbox')
        x1=int(bbox.find('xmin').text)
        y1=int(bbox.find('ymin').text)
        x2=int(bbox.find('xmax').text)
        y2=int(bbox.find('ymax').text)
        print(x1,y1,x2,y2)
        coor.append([x1,y1,x2,y2])
    return coor
