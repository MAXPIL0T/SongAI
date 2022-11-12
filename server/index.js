import logger from 'morgan';
import express from 'express';
import {spawn} from 'child_process';
import fs from 'fs';

const app = express();
const port = process.env.PORT || 3000;

app.use(logger('dev'));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use('/', express.static('client'));

app.post('songToText', async (req, res) => {
  try {
    let file_name = req.files.upload.name;
    await req.files.upload.mv(`./server/tmp/${file_name}`, (error) =>      
    console.log(error));
  }  catch (error) {
    console.log(error);
    res.status(500).send('There was an error, please try again.');
  }
});

app.get('py', (req, res) => {
     let dataToSend;

     const python = spawn('python3', ['server/python/script.py', "hi", "Duyen"]);

     python.stdout.on('data', function (data) {
      dataToSend = data.toString();
     });

     python.stderr.on('data', data => {
      console.error(`stderr: ${data}`);
     });

     python.on('exit', (code) => {
     console.log(`child process exited with code ${code}, ${dataToSend}`);
     response.send(dataToSend);
    }); 
});

app.listen(port, () => {
  console.log(`Server started on http://localhost:${port}`);
});
