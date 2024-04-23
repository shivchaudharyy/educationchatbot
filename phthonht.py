import os
import json
from flask import Flask, request

app = Flask(__name__)

@app.route('/create_folder', methods=['POST'])
def create_folder():
    data = request.json
    folder_name = data['folderName']
    # Replace 'path_to_your_folder' with the actual path where you want to create the folder
    folder_path = os.path.join('path_to_your_folder', folder_name)
    try:
        os.makedirs(folder_path)
        return json.dumps({'success': True}), 200, {'ContentType':'application/json'} 
    except OSError as e:
        return json.dumps({'success': False, 'error': str(e)}), 400, {'ContentType':'application/json'} 

if __name__ == '__main__':
    app.run(debug=True)
