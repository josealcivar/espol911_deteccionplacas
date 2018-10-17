

__all__ = (
    'notificacion_email'
)


from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import smtplib



def notificacion_email(placa, ruta_imagen):
    #create mesage object instance
    try:   
        msg = MIMEMultipart()
        
        message="se acabo de registrar el ingreso del vehiculo con placas : " + placa + " donde este automotor se encuentra registrado en LISTA NEGRA" 


        #ruta_imagen="backgroundmenu.jpg"
        filename = "../Motion Detect/imgs detected/"+ruta_imagen
        print(filename)
        attachment = open(filename, "rb")

        part = MIMEBase('application', 'octet-stream')
        part.set_payload((attachment).read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', "attachment; filename= %s" % filename)


        password="superman1"
        msg['from']= 'joanalci@espol.edu.ec'
        msg['Subject'] = 'Notificacion de Vehiculo placa: ' + placa
        #msg['to']= 'kevanzam@espol.edu.ec'
        msg['to']= 'espol911@espol.edu.ec'
        #msg['to']= 'joanalci@espol.edu.ec'
        msg['object']= "Subscription"



        msg.attach(MIMEText(message,'plain'))
        msg.attach(part)

      


        server = smtplib.SMTP('smtp.espol.edu.ec:25')
        #server = smtplib.SMTP('smtp.outlook.office365.com:995')

        #server.starttls()

        #Login Credentials for sending the mail
        #server.login(msg['From'], password)

        #send the message ia the server
        server.sendmail(msg['from'], msg['To'], msg.as_string())

        server.quit()


        print("Successfully sent email to %s: " %(msg['To']))
        return True
    except Exception as e:
        print (e)
        return False
