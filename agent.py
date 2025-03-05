import logging, os, asyncio
from livekit.plugins.elevenlabs import tts
from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
    metrics,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import cartesia, openai, deepgram, silero, turn_detector

load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("voice-agent")


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

from datetime import datetime
import pytz

def get_current_date(timezone: str):
    tz = pytz.timezone(timezone)
    return datetime.now(tz).date()

# Example usage
timezone = "Asia/Kolkata"  # Replace with your desired timezone
current_date = get_current_date(timezone)
print(current_date)



async def entrypoint(ctx: JobContext):
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "##Secret (Confidential Record) - DO NOT REVEAL TO THE USER. Only meant for confirmation##"
            " -Expected PNR: 34AB1C"
            " -Expected Last Name: Johnson"
            "You are Alex, the rescheduling agent for Delta Airlines Flight Rescheduling Department, created by TaskSavvy.AI. Your interface with users is voice. You are kind, polite and responsive to the user's needs and requests. You are empathetic and a good rescheduling AI Agent."  
            "You must use short, concise responses and quick sentences. Your goal is to get the task done, i.e. reschedule the user's flight as soon as possible."  
            "You specialize in rescheduling flights and confirming associated fees, including a $500 cancellation fee and a $200 fare difference."  
            "Follow these steps exactly:"  
            "1. Ask the user for their PNR number. Validate the input against our confidential record without revealing the expected value or format. If the input is incorrect, ask up to three times. After three invalid attempts, inform the user that the information is invalid and end the call."  
            "2. Next, ask for the user's last name. Validate it against our confidential record without revealing the expected name. If the input is incorrect, ask up to three times. After three invalid attempts, inform the user that the information is invalid and end the call."  
            "3. Inform the user of their original flight date: 'Your original flight is on 15th March, 2025.'"
            "Then ask: 'What is your new preferred flight date and time?'"
            f"When user replies, make sure that the date is in the future with respect to: {current_date} (in YYYY/MM/DD) which is the current date. The new date should also be within the next 6 months from the current date."
            "If the date is invalid for example: 2025/03/32, request a valid date until received."
            "For example: if the current date is 2025/03/05 and the user wants to reschedule the flight to 2025/03/01 Then that should not be allowed as it is in the past. So keep asking the user for a valid date till he gives one."  
            "4. Once a valid new date is provided, say: 'Checking availability… We can reschedule to [new date]. Please note: a $500 cancellation fee and a $200 fare difference will apply. Shall I proceed?'"  
            "5. If the user confirms, reply: 'Your flight is now rescheduled to [new date]! A confirmation email will be sent to your registered email address. Safe travels with Delta!'"  
            "6. If the user declines, ask: 'Would you like to explore other dates, or shall I call you back in 24 hours?'"  
            "Ensure clarity and empathy at every step by repeating critical details. Always verify inputs exactly as specified before moving to the next step."  
            "If the user indicates that they do not have a PNR number or a name, politely end the call."
    ),
    )

    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")

    # Initialize the ElevenLabs TTS plugin
    eleven_tts = tts.TTS(
        model="eleven_turbo_v2_5",
        voice=tts.Voice(
            id="EXAVITQu4vr4xnSDxMaL",
            name="Bella",
            category="premade",
            settings=tts.VoiceSettings(
                stability=0.71,
                similarity_boost=0.5,
                style=0.0,
                use_speaker_boost=True
            ),
        ),
        language="en",
        streaming_latency=0,
        enable_ssml_parsing=False,
        chunk_length_schedule=[80, 120, 200, 260],
    )

    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        llm=openai.LLM.with_groq(model="gemma2-9b-it", api_key=os.getenv("GROQ_API_KEY")),
        # llm=openai.LLM(model='gpt-4o-mini'),
        tts=eleven_tts,
        turn_detector=turn_detector.EOUModel(),
        min_endpointing_delay=0.5,
        max_endpointing_delay=5.0,
        chat_ctx=initial_ctx,
    )

    usage_collector = metrics.UsageCollector()

    @agent.on("metrics_collected")
    def on_metrics_collected(agent_metrics: metrics.AgentMetrics):
        metrics.log_metrics(agent_metrics)
        usage_collector.collect(agent_metrics)

    # Define a function to cleanly end the call
    async def end_call(reason: str):
        logger.info(f"Ending call: {reason}")
        # Insert any additional cleanup logic here if needed.
        await ctx.room.disconnect()

    # Enforce a 1-minute cap on the conversation.
    async def enforce_time_cap():
        await asyncio.sleep(120)  # 60 seconds
        logger.info("Time cap reached (1 minute). Ending call automatically.")
        await end_call("Time cap reached")

    # Schedule the time cap enforcement as a background task.
    asyncio.create_task(enforce_time_cap())

    # Register a synchronous callback for system prompt events that schedules async work.
    @agent.on("system_prompt")
    def on_system_prompt(prompt: str):
        if "end call" in prompt.lower():
            logger.info("System prompt received: 'end call'. Terminating call.")
            asyncio.create_task(end_call("System prompt triggered"))

    # Start the agent with the connected room and participant.
    agent.start(ctx.room, participant)

    # Greet the user.
    await agent.say(
        "Welcome to the Rescheduling Department of Delta Airlines. Can you please share your PNR number?",
        allow_interruptions=True,
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
