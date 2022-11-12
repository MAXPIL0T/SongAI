from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
import sys

vocal_path = sys.argv[1]
genre = sys.argv[2]

video_clip = VideoFileClip('/videos/{}.mp4'.format(genre))
genre_audio = AudioFileClip("/beats/{}.mp3".format(genre))
vocal_audio = AudioFileClip(vocal_path)

genre_clip = genre_audio.volumex(0.75)
vocal_clip = vocal_audio.volume(1.0)

final_audio = CompositeAudioClip([genre_clip, vocal_clip])

final_clip = video_clip.set_audio(final_audio)
final_clip.write_videofile("final.mp4")

print()
sys.stdout.flush()