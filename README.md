![tag:innovationlab](https://img.shields.io/badge/innovationlab-3D8BD3)
![tag:hackathon](https://img.shields.io/badge/hackathon-5F43F1)

🚨 Dispatch AI

Your AI-powered emergency response assistant. When a 911 call comes in, a video link is automatically sent to the caller's phone. The caller points their camera at the scene, and our multi-agent system analyzes the live feed in real time — identifying the emergency type, assessing severity, and producing a structured dispatch brief for responders. No app download required. No dispatch center integration needed.

Agent Name: Triagealertagent
Agent Address: agent1q2fu34yahnp7k8t0kyk44kwhn2jlmhe4vl9ljr5z0e93zmh9q5kgz3fmn2v

What I Can Do

🎥 Live Video Analysis: The caller receives a link and points their camera at the scene. Frames are analyzed every few seconds by a vision agent that identifies injuries, fire, hazards, and number of people.

🧠 Intelligent Triage: Based on the visual feed, the agent determines the nature and severity of the emergency — medical, fire, violence, or other — and how urgent a response is needed.

🚒 Smart Dispatch Recommendations: The dispatch coordinator agent decides what resources are needed — how many fire trucks, whether police are required, or if an ambulance alone is sufficient.

📋 Structured Incident Brief: All findings are compiled into a CAD-compatible summary that any dispatch center can read, with zero integration required on their end.

🌐 No App, No Barrier: The civilian just taps a link. Everything else is handled by the agents on the backend.

How It Works

A 911 call comes in
The caller receives an SMS link and taps it to open the video session
They point their camera at the scene
The vision agent analyzes frames every few seconds
The triage agent converts visual data into a priority report
The dispatch coordinator produces a structured brief
The dispatcher sees the brief populate in real time — no manual input needed


Demo note: The full system is designed to send an automatic SMS to the caller's phone the moment a 911 call comes in. Due to carrier regulations, US phone number SMS approval (A2P 10DLC) requires several days of processing time and could not be completed within the hackathon window. For this demo, the SMS link is delivered via a QR code displayed on the dispatcher dashboard — the caller scans it and the experience is identical from that point on. The SMS infrastructure (sms_trigger.py) is fully built and ready to activate once the number is approved.



Agent Breakdown

AgentRoleIntake AgentSends the SMS/QR link, opens the video sessionVision AgentAnalyzes frames — tags injuries, fire, hazards, people countTriage AgentConverts visual data into a structured priority reportDispatch CoordinatorDecides resources needed and formats the final brief

Example Interaction

Caller clicks on the link and points camera at a house fire

Vision Agent detects:

fire_visible:     true
smoke_visible:    true
people_count:     2
injury_visible:   true (moderate)
hazards:          structural collapse risk
confidence:       91%

Dispatch Brief produced:

🚨 INCIDENT REPORT — AUTO-GENERATED
Type:         Structure Fire with Casualties
Severity:     HIGH
Recommended:  2x Fire Truck, 1x Ambulance, 1x Police Unit
People:       2 visible, 1 showing signs of injury
Hazards:      Smoke inhalation risk, possible structural collapse
Timestamp:    2026-06-20T18:45:00Z

What Makes This Different

Our system requires nothing from the dispatch center — the agent outputs a standard CAD-compatible summary that any center can read. The innovation is pushing intelligence to the civilian side so the infrastructure barrier disappears entirely.

Example Queries


"What is happening at the scene?"
"How many responders are needed?"
"Is this a medical emergency or a fire?"
"What hazards are present?"
"Generate a dispatch brief for this incident"