const { Client, Intents } = require('discord.js');
const config = require('./token.json');
const client = new Client({ intents: [Intents.FLAGS.GUILDS, Intents.FLAGS.GUILD_MESSAGES] });
const token = config.token;

client.on('ready', () => {
  console.log(`Logged in as ${client.user.tag}!`);
});

client.on('messageCreate', (message) => {
    if (message.content.startsWith('!play')) {
        const voiceChannel = message.member.voice.channel;
    
        if (!voiceChannel) {
          return message.reply('Veuillez rejoindre un salon vocal avant d utiliser cette commande.');
        }
    
        const connection = joinVoiceChannel({
          channelId: voiceChannel.id,
          guildId: voiceChannel.guild.id,
          adapterCreator: voiceChannel.guild.voiceAdapterCreator,
        });
    
        const stream = ytdl('YOUR_YOUTUBE_VIDEO_URL', { filter: 'audioonly' });
        const resource = createAudioResource(stream, { inputType: 1 });
        const player = createAudioPlayer({ behaviors: { noSubscriber: NoSubscriberBehavior.Pause } });
    
        connection.subscribe(player);
        player.play(resource);
    
        message.reply('La musique est en train d\'être jouée !');
      }
});

client.login(token);