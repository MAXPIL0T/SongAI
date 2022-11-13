import gTTS from 'gtts';
     
function tts (input) {
    let gtts = new gTTS(input, 'en');
    const file_name = randomFileName();
    
    gtts.save(`./server/sound_files/${file_name}.mp3`, function (err, result){
        if(err) { throw new Error(err); }
    });
    return `${file_name}.mp3`
}

function randomFileName() {
    return new Date().getTime().toString();
}

export default tts;