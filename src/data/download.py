import requests, zipfile, io
 
def download (url, save_path,):
   r = requests.get(url)
   z = zipfile.ZipFile(io.BytesIO(r.content))
   z.extractall(filepath)
 
if __name__ == '__main__':
   url = 'www.cis.um.edu.mo/research/shelldataset//static/new_shell_images_2nd.zip'
   filepath = '../../data'
 
   download(url, filepath)
