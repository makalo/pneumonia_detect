import xml.etree.ElementTree as ET
def read_xml(image_name):
    path='/home/makalo/workspace/code/kaggle/Faster-RCNN_TF/data/VOCdevkit2007/VOC2007/Annotations/'
    tree=ET.parse(path+image_name)
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
if __name__=='__main__':
    read_xml('0b1d5b23-c3ab-41b5-b82d-a16a3d6b5674.xml')