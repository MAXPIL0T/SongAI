import logger from 'morgan';
import express from 'express';
import {spawn} from 'child_process';
import fs from 'fs';
import multer from 'multer';
import path from 'path';
import tts from './textToSpeech.js'

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

app.post('/textToSpeech', (req, res) => {
  let {text} = req.body;
  let file_name = tts(text);
  console.log(file_name)
  res.send({
    str: text,
    file_name: file_name
  });
});

app.post("/uploadwav", upload.single("audio_data"), function(req,res) {
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
    res.send({
      str: res_str,
      file_name: req.file.originalname
    });
   }); 
});


app.post('/getMusicVideo', (req, res) => {
  let {text, name} = req.body;
  const file_path = path.resolve(`./server/sound_files/${name}`)
  const new_file_name = `${name.substring(0, name.length - 4)}.mp4`;

  const python = spawn('python3', ['server/python/musicVideoMaker.py', file_path, text, new_file_name]);
  let video_file_path;

  python.stdout.on('data', data => {
    video_file_path = data.toString();
  });

  python.stderr.on('data', data => {
    console.error(`stderr: ${data}`);
  });

  python.on('exit', (code) => {
    console.log(video_file_path);
    res.send(video_file_path);
  }); 
});

app.get('/content/:id', (res, req) => {
  video_id = req.params.id
  // res.file
});

app.listen(port, () => {
  console.log(`Server started on http://localhost:${port}`);
});