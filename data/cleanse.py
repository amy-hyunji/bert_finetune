form = ['train', 'dev', 'test']

for _form in form:
   fi = open(f'{_form}.csv', 'rb')
   data = fi.read()
   fi.close()
   fo = open(f'{_form}.csv', 'wb')
   fo.write(data.replace(b'\x00', b''))
   fo.close()
