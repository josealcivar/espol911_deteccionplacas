#
# Copyright (c) 2018 Rafael Rivadeneira
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
#
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.


__all__ = (
    'detect',
    'post_process',
    'detectLicensePlate',
    'consulta_Listanegra'
)

import collections
import math
import time
import matplotlib.pyplot as plt
import sys

import cv2
import numpy
import tensorflow as tf

import common
import model

from sendemail import notificacion_email

import urllib
from bs4 import BeautifulSoup
import mysql.connector

con=mysql.connector.connect(host='localhost', user='root', password='root', database='anpr_espol911')
mycursor=con.cursor()


def make_scaled_ims(im, min_shape):
    ratio = 1. / 2 ** 0.5
    shape = (im.shape[0] / ratio, im.shape[1] / ratio)

    while True:
        shape = (int(shape[0] * ratio), int(shape[1] * ratio))
        if shape[0] < min_shape[0] or shape[1] < min_shape[1]:
            break
        yield cv2.resize(im, (shape[1], shape[0]))


def consulta_Listanegra(placa, name_image):
    try:
        mycursor.execute("SELECT * FROM vehiculo_lista_negra_vehiculos l, vehiculo_vehiculo v WHERE  l.vehiculo_id = v.id AND v.placa = %s", (placa, ))
        myresult = mycursor.fetchall()
        if len(myresult)>0: # verifica que el vehiculo se encuentre registrado en la base de datos 
           notificacion_email(placa, name_image) # se envia placa y ruta de imagen para mostrar en email
           
    except Exception as e:
        print("No pudo consultar lista negra")  
        print (e)      
        
'''     
    @description: registra en la base de datos la placa y caracteristica de vehiculos
                  exista o no datos de atm, igual registra la placa detectada.
'''

def agregar_database(placa, feature, fecha, hora, name_file):
    try:
        camara = "INGRESO"
        # placa ='GSZ5800'    
        mycursor.execute("SELECT * FROM vehiculo_vehiculo WHERE placa = %s", (placa, ))
        myresult = mycursor.fetchall()
     
        if len(myresult)>0: # obtiene una longitud > 0 (solo es para verificar si hay registro)
            id_row = myresult[0][0]  # obtiene el id del vehiculo
            
            print("PRIMERO")
            print(id_row)
            mycursor.execute("INSERT INTO vehiculo_flujo_vehicular (fecha, horacaptura, camara, vehiculo_id, rutaimagen) VALUES (%s, %s, %s, %s, %s)", (fecha, hora, camara, id_row, name_file))
            con.commit()
            print("1 record inserted, ID:", mycursor.lastrowid)
            
        elif len(feature)==1: # verifica si el arreglo contiene solo registro de placa y no de atm (dado el caso que la placa registrada no existe y no encuentra datos de atm) guarda solo la placa
            print("SEGUNDO")
            mycursor.execute("INSERT INTO vehiculo_vehiculo (placa) VALUES (%s)", feature) 
            id_row = mycursor.lastrowid
            
            mycursor.execute("INSERT INTO vehiculo_flujo_vehicular (fecha, horacaptura, camara, vehiculo_id, rutaimagen) VALUES (%s, %s, %s, %s, %s)", (fecha, hora, camara, id_row, name_file))
            con.commit()
            
        else: # guarda un nuevo vehiculo y su registro de atm con la hora de captura y fecha
            print("TERCERO")
            print(feature)
            mycursor.execute("INSERT INTO vehiculo_vehiculo (placa, marca, color, anio_matricula, modelo, clase, fecha_matricula, anio_vehiculo, servicio, fecha_caducidad) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", feature) 
            id_row = mycursor.lastrowid # obtiene el id del registro del vehiculo
            print("1 record inserted, ID:", mycursor.lastrowid)
            mycursor.execute("INSERT INTO vehiculo_flujo_vehicular (fecha, horacaptura, camara, vehiculo_id, rutaimagen) VALUES (%s, %s, %s, %s, %s)", (fecha, hora, camara, id_row, name_file))
            con.commit()
            print("2 record inserted, ID:", mycursor.lastrowid)
            
        
        return True
    except:
        print("algo paso")
        return False


def consultaPlaca(placa):
    data_vehicle=[]
    data_vehicle.append(placa)
    
    try:
        quote_page = 'http://consultas.atm.gob.ec/PortalWEB/paginas/clientes/clp_grid_citaciones.jsp?ps_tipo_identificacion=PLA&ps_identificacion='+ placa +'&ps_placa='
        page = urllib.request.urlopen(quote_page)
        soup = BeautifulSoup(page, 'html.parser')
        table = soup.find('table',{'cellpadding':"2"})
        rows = table.find_all('tr')
        for tr in rows:
            for wrapper in tr.find_all(lambda tag: tag.name == 'td' and tag.get('class') == ['detalle_formulario'] and tag.get('class') == ['detalle_formulario']):
                data_vehicle.append(wrapper.text)
                print (wrapper.text)

        print(data_vehicle)
        return data_vehicle
    except:
        print('Datos no disponibles')
        return data_vehicle


def detect(im, param_vals):
    """
    Detect all bounding boxes of number plates in an image.
    :param im:
        Image to detect number plates in.
    :param param_vals:
        Model parameters to use. These are the parameters output by the `train`
        module.
    :returns:
        Iterable of `bbox_tl, bbox_br, letter_probs`, defining the bounding box
        top-left and bottom-right corners respectively, and a 7,36 matrix
        giving the probability distributions of each letter.
    """

    # Convert the image to various scales.
    scaled_ims = list(make_scaled_ims(im, model.WINDOW_SHAPE))

    # Load the model which detects number plates over a sliding window.
    x, y, params = model.get_detect_model()

    # Execute the model at each scale.
    with tf.Session(config=tf.ConfigProto()) as sess:
        y_vals = []

        for scaled_im in scaled_ims:
            feed_dict = {x: numpy.stack([scaled_im])}
            feed_dict.update(dict(zip(params, param_vals)))
            y_vals.append(sess.run(y, feed_dict=feed_dict))
            #plt.imshow(scaled_im) # esto va comentado con comillas
            #plt.show()
    writer = tf.summary.FileWriter("logs/", sess.graph)

    # Interpret the results in terms of bounding boxes in the input image.
    # Do this by identifying windows (at all scales) where the model predicts a
    # number plate has a greater than 50% probability of appearing.
    #
    # To obtain pixel coordinates, the window coordinates are scaled according
    # to the stride size, and pixel coordinates.
    count_detect = 0
    for i, (scaled_im, y_val) in enumerate(zip(scaled_ims, y_vals)):
        for window_coords in numpy.argwhere(y_val[0, :, :, 0] >
                                            -math.log(1. / 0.99 - 1)):
            letter_probs = (y_val[0,
                            window_coords[0],
                            window_coords[1], 1:].reshape(
                7, len(common.CHARS)))
            letter_probs = common.softmax(letter_probs)

            img_scale = float(im.shape[0]) / scaled_im.shape[0]

            bbox_tl = window_coords * (8, 4) * img_scale
            bbox_size = numpy.array(model.WINDOW_SHAPE) * img_scale

            present_prob = common.sigmoid(
                y_val[0, window_coords[0], window_coords[1], 0])
            count_detect += 1
            yield bbox_tl, bbox_tl + bbox_size, present_prob, letter_probs
            '''print("count detect:", count_detect)
            print("show return window: ", bbox_tl, "return windows box: ", bbox_tl + bbox_size)
            print("present: ", present_prob)
            print("letter: ", letter_probs_to_code(letter_probs))'''


def _overlaps(match1, match2):
    bbox_tl1, bbox_br1, _, _ = match1
    bbox_tl2, bbox_br2, _, _ = match2
    return (bbox_br1[0] > bbox_tl2[0] and
            bbox_br2[0] > bbox_tl1[0] and
            bbox_br1[1] > bbox_tl2[1] and
            bbox_br2[1] > bbox_tl1[1])


def _group_overlapping_rectangles(matches):
    matches = list(matches)
    num_groups = 0
    match_to_group = {}
    for idx1 in range(len(matches)):
        for idx2 in range(idx1):
            if _overlaps(matches[idx1], matches[idx2]):
                match_to_group[idx1] = match_to_group[idx2]
                break
        else:
            match_to_group[idx1] = num_groups
            num_groups += 1

    groups = collections.defaultdict(list)
    for idx, group in match_to_group.items():
        groups[group].append(matches[idx])

    return groups


def post_process(matches):
    """
    Use non-maximum suppression on the output of `detect` to filter.
    Take an iterable of matches as returned by `detect` and merge duplicates.
    Merging consists of two steps:
      - Finding sets of overlapping rectangles.
      - Finding the intersection of those sets, along with the code
        corresponding with the rectangle with the highest presence parameter.
    """
    groups = _group_overlapping_rectangles(matches)

    for group_matches in groups.values():
        mins = numpy.stack(numpy.array(m[0]) for m in group_matches)
        maxs = numpy.stack(numpy.array(m[1]) for m in group_matches)
        present_probs = numpy.array([m[2] for m in group_matches])
        letter_probs = numpy.stack(m[3] for m in group_matches)

        yield (numpy.max(mins, axis=0).flatten(),
               numpy.min(maxs, axis=0).flatten(),
               numpy.max(present_probs),
               letter_probs[numpy.argmax(present_probs)])


def letter_probs_to_code(letter_probs):
    return "".join(common.CHARS[i] for i in numpy.argmax(letter_probs, axis=1))


def detectLicensePlate(crop_img, pv):

    im_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY) / 255.

    for pt1, pt2, present_prob, letter_probs in post_process(
            detect(im_gray, pv)):
        pt1 = tuple(reversed(list(map(int, pt1))))
        pt2 = tuple(reversed(list(map(int, pt2))))

        code = letter_probs_to_code(letter_probs)


    try:
        print ("\nBest accurate detection: ", code)
        fecha       = time.strftime("%Y-%m-%d", time.localtime())
        hora        = time.strftime("%H:%M:%S", time.localtime())

        fechalabel  = time.strftime("%Y_%m_%d", time.localtime())
        horalabel   = time.strftime("%Hh%Mm%Ss", time.localtime())
        name_file=code+'_'+fechalabel+'_'+horalabel+'.jpg'
        print(name_file)
        print(len(name_file))
       
        data_vehicle=consultaPlaca(code)
        agregar_database(code, data_vehicle, fecha, hora, name_file)
        
        print(name_file)
        return name_file, code

    except:
        
        code='1'
        print ("\nNo License Plate Detected")
        return code, code
