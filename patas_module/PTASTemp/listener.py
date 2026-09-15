import socket

def listen_on_port(port=5000):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind(('127.0.0.1', port))
        server_socket.listen()
        print(f"Listening on port {port}...")

        while True:
            client_socket, address = server_socket.accept()
            with client_socket:
                print(f"Connected by {address}")
                while True:
                    data = client_socket.recv(1024)
                    if not data:
                        break
                    print("Received:", data.decode())

# Start listening on port 5000
listen_on_port()
