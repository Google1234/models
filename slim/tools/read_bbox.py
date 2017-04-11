import numpy
import os
def read_from_xml(class_name,image_name,root="/ssd/lidenghui/jt/fishes/train_box"):
    '''
    :param class_name:
    :param image_name:
    :return:
    file not exist or image do not contain specific class_name bbox ,return [0,0,1,1]
    else return ["ymin", "xmin", "ymax", "xmax"]
    '''
    if class_name=="NoF":
        return 1,[0,0,1,1]
    image_name+='.xml'
    file_name=os.path.join(root,class_name,image_name)
    #print ('############',file_name)
    bboxs = []
    if os.path.exists(file_name)==False:
        print (file_name,'bbox file not exist')
        return 0,[0,0,1,1]

    import xml.etree.ElementTree as ET
    tree = ET.parse(file_name)
    objs = tree.getroot().findall("object")
    size= tree.getroot().findall("size")
    width=int(size[0].find("width").text)
    height = int(size[0].find("height").text)
    bbox_order=["ymin", "xmin", "ymax", "xmax"]
    ratio_order=[1.0/height,1.0/width,1.0/height,1.0/width]
    for obj in objs:
        if obj.find('name').text==class_name:
            box=obj.find("bndbox")
            bboxs.append([int(box.find(s).text) for s in bbox_order])
    #if len(bboxs)==0:
    #    bboxs.append([0,0,int(tree.find("size").find("height").text),int(tree.find("size").find("width").text)])  ######bbox_order=["ymin", "xmin", "ymax", "xmax"]
    #print bboxs
    if len(bboxs)==0:
        print (file_name,'bbox file not contain this class bbox')
        return 0,[0,0,1,1]
    #print (file_name)
    #print ([bboxs[i][j]*ratio_order[j] for i in range(len(bboxs)) for j in range(4)])
    return len(bboxs),[bboxs[i][j]*ratio_order[j] for i in range(len(bboxs)) for j in range(4)]


def Statistics():
    #**********************************#
    #Statistics Image length and width #
    '''
    ALB  average width: 210.132370638
    ALB  max width: 698
    ALB  average height: 142.381468111
    ALB  max height: 431
    BET  average width: 283.102941176
    BET  max width: 529
    BET  average height: 206.977941176
    BET  max height: 489
    DOL  average width: 255.75
    DOL  max width: 592
    DOL  average height: 135.612903226
    DOL  max height: 273
    LAG  average width: 297.147058824
    LAG  max width: 487
    LAG  average height: 245.421568627
    LAG  max height: 414
    OTHER  average width: 202.092857143
    OTHER  max width: 535
    OTHER  average height: 154.30952381
    OTHER  max height: 306
    SHARK  average width: 281.606936416
    SHARK  max width: 671
    SHARK  average height: 161.61849711
    SHARK  max height: 319
    YFT  average width: 275.501179245
    YFT  max width: 826
    YFT  average height: 187.886792453
    YFT  max height: 443
    '''
    dir_path="/Users/jt/code/MachineLearning/fish_data/anotation/"
    class_names=["ALB","BET","DOL","LAG","OTHER","SHARK","YFT"]
    bbox_order=["ymin", "xmin", "ymax", "xmax"]

    import os
    import xml.etree.ElementTree as ET

    for class_name in class_names:
        bboxs = []
        for filename in os.listdir(os.path.join(dir_path,class_name)):
            #print filename
            tree = ET.parse(os.path.join(dir_path,class_name,filename))
            objs = tree.getroot().findall("object")
            for obj in objs:
                if obj.find('name').text==class_name:
                    box=obj.find("bndbox")
                    bboxs.append([int(box.find(s).text) for s in bbox_order])
            #if len(bboxs)==0:
            #    bboxs.append([0,0,int(tree.find("size").find("height").text),int(tree.find("size").find("width").text)])  ######bbox_order=["ymin", "xmin", "ymax", "xmax"]
        import numpy as np
        bboxs=np.array(bboxs)
        width=bboxs[:,3]-bboxs[:,1]
        height = bboxs[:, 2] - bboxs[:, 0]
        print class_name," average width:",width.mean()
        print class_name, " max width:", width.max()
        print class_name," average height:",height.mean()
        print class_name, " max height:", height.max()


