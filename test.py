from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def beranda():
    return render_template('Beranda.html')

@app.route('/interpretasi')
def interpretasi():
    return render_template('Interpretasi.html')

@app.route('/tentang')
def tentang():
    return render_template('Tentang.html')

@app.route('/lainnya')
def lainnya():
    return render_template('Lainnya.html')

if __name__ == '__main__':
    app.run(debug=True)