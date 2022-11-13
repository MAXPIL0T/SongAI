let rec = null;
let audioStream = null;

const recordButton = document.getElementById("recordButton");
const transcribeButton = document.getElementById("transcribeButton");
const textEntryButton = document.getElementById('text-entry');

textEntryButton.addEventListener('click', textInput);
recordButton.addEventListener("click", startRecording);
transcribeButton.addEventListener("click", transcribeText);

function textInput() {
    document.getElementById('content').innerHTML = `
        <textarea name="text-entry-field" id="text-entry-field"></textarea>
        <button id="submit" and generate video></button>"
        <div id='music-vid-c'></div>
    `;

    document.getElementById('submit').addEventListener('click', async function() {
        const res = await fetch('/textToSpeech', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({text: document.getElementById('text-entry-field').value})
        });
        const text = await res.text();
        console.log(text)
        await musicVideoConfirmation(text);

    });
}

function startRecording() {

    let constraints = { audio: true, video:false }

    recordButton.disabled = true;
    transcribeButton.disabled = false;

    navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
        const audioContext = new window.AudioContext();
        audioStream = stream;
        const input = audioContext.createMediaStreamSource(stream);
        rec = new Recorder(input, { numChannels: 1 })
        rec.record()
        document.getElementById("output").innerHTML = "Recording started..."
    }).catch(function(err) {
        recordButton.disabled = false;
        transcribeButton.disabled = true;
    });
}

function transcribeText() {
    document.getElementById("output").innerHTML = "Converting audio to text..."
    transcribeButton.disabled = true;
    recordButton.disabled = false;
    rec.stop();
    audioStream.getAudioTracks()[0].stop();
    rec.exportWAV(uploadSoundData);
}

function uploadSoundData(blob) {
    const filename = "sound-file-" + new Date().getTime() + ".wav";
    const formData = new FormData();
    formData.append("audio_data", blob, filename);
    
    fetch('/uploadwav', {
        method: 'POST',
        body: formData
    }).then(async result => { 
        const text = await result.text();
        document.getElementById("output").innerHTML = text;
        await musicVideoConfirmation(text);
    }).catch(error => { 
        document.getElementById("output").innerHTML = "An error occurred: " + error;
    })
}

async function musicVideoConfirmation(text) {
    let parsed = JSON.parse(text);
    const step_div = document.getElementById('music-vid-c');
    step_div.innerHTML = `
        <button id="getVideo">Generate Music Video<\button>
        
    `;
    const btn = document.getElementById('getVideo');
    btn.addEventListener('click', async function() {
        const video_url = await fetch("/getMusicVideo", {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({text: parsed.str, name: parsed.file_name})
        });
        // document.getElementById('video-box').innerHTML = `
        //     <video src="./content/${await video_url.text()}"></video>
        // `;
    });
}