let rec = null;
let audioStream = null;

const recordButton = document.getElementById("recordButton");
const transcribeButton = document.getElementById("transcribeButton");

recordButton.addEventListener("click", startRecording);
transcribeButton.addEventListener("click", transcribeText);

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
        document.getElementById("output").innerHTML = await result.text();
    }).catch(error => { 
        document.getElementById("output").innerHTML = "An error occurred: " + error;
    })
}