import os
from io import BytesIO
import tempfile
from pathlib import Path

# Setup mock env before importing app
os.environ["SECRET_KEY"] = "test"
os.environ["APP_PASSWORD"] = "testpass"
os.environ["TMP_DIR"] = tempfile.gettempdir()

from app import app, TMP_BASE

def run_tests():
    app.testing = True
    client = app.test_client()
    
    print("1. Login...")
    # Login to get session
    res = client.post('/login', data={'password': 'testpass'})
    if res.status_code not in (302, 200):
        print(f"Login failed! Status: {res.status_code}")
        return
        
    print("Login successful.")

    # Create a dummy image
    dummy_image = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    
    print("\n2. Uploading valid file...")
    data = {
        'photos': (BytesIO(dummy_image), 'test_image.png')
    }
    
    res = client.post('/bg/upload', data=data, content_type='multipart/form-data')
    print(f"Upload Response Status: {res.status_code}")
    print(f"Upload Response Body: {res.get_json()}")
    
    # Check if file was saved correctly
    res_json = res.get_json()
    if res_json and 'accepted' in res_json and len(res_json['accepted']) > 0:
        file_id = res_json['accepted'][0]['id']
        print(f"\n3. Checking if file {file_id} exists on disk...")
        
        # We need to find the session ID from cookies to construct the path
        # But we can just search TMP_BASE for the file_id
        found = False
        for p in Path(TMP_BASE).rglob(file_id):
            print(f"Found file at: {p}")
            found = True
            print(f"File size on disk: {p.stat().st_size} bytes")
            break
            
        if not found:
            print("ERROR: File not found on disk!")
            
    # Test file that is too large (using mock content_length or large string)
    print("\n4. Uploading large file (should fail/reject)...")
    # Generating 21MB string (MAX_FILE_MB is 20)
    large_data = b"0" * (21 * 1024 * 1024)
    data_large = {
        'photos': (BytesIO(large_data), 'large_image.png')
    }
    
    res = client.post('/bg/upload', data=data_large, content_type='multipart/form-data')
    print(f"Upload Large File Status: {res.status_code}")
    print(f"Upload Large File Body: {res.get_json()}")
    
if __name__ == '__main__':
    run_tests()
