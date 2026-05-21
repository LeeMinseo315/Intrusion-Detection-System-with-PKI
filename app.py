from flask import Flask
import ssl

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello mTLS"

if __name__ == "__main__":
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain(
    certfile="server_chain.crt",
    keyfile="server.key"
    )
    context.load_verify_locations(cafile="ca.crt")
    context.verify_mode = ssl.CERT_REQUIRED

    app.run(host="0.0.0.0", port=5000, ssl_context=context)
