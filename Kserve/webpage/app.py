import subprocess

from flask import Flask, render_template
app = Flask(__name__)

def run_command(command):
    return subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).stdout.read()

@app.route('/<command>')
def command_server(command):
    return render_template('index.html', a = run_command(command).decode('ascii'))

if __name__ == '__main__':
    app.run()
