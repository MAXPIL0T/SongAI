import logger from 'morgan';
import express from 'express';
import {spawn} from 'child_process';
import fs from 'fs';
import multer from 'multer';
import path from 'path';

const storage = multer.diskStorage(
    {
        destination: './server/sound_files/',
        filename: function (req, file, cb ) {
            cb( null, file.originalname);
        }
    }
);

const upload = await multer( { storage: storage } );


const app = express();
const port = process.env.PORT || 3000;

app.use(logger('dev'));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use('/', express.static('client'));

app.post("/uploadwav", upload.single("audio_data"), function(req,res){
  const file_path = path.resolve(`./server/sound_files/${req.file.originalname}`);

  const python = spawn('python3', ['server/python/speechToText.py', file_path]);
  let res_str;

  python.stdout.on('data', data => {
    res_str = data.toString();
  });

  python.stderr.on('data', data => {
    console.error(`stderr: ${data}`);
   });

  python.on('exit', (code) => {
    console.log(res_str)
    res.send(res_str);
   }); 
});


app.get('/py', (req, res) => {
     let dataToSend;

     const python = spawn('python3', ['server/python/script.py', "hi", "Duyen"]);

     python.stdout.on('data', function (data) {
      dataToSend = data.toString();
     });

     python.stderr.on('data', data => {
      console.error(`stderr: ${data}`);
     });

     python.on('exit', (code) => {
     res.send(res);
    }); 
});

app.listen(port, () => {
  console.log(`Server started on http://localhost:${port}`);
});