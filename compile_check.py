import py_compile
import glob

files = ['app.py'] + glob.glob('src/*.py') + glob.glob('tests/*.py')
for path in files:
    try:
        py_compile.compile(path, doraise=True)
        print(f'OK {path}')
    except Exception as exc:
        print(f'ERROR {path} -> {exc}')
