Emergency Vehicle Detection & Signal Time Management System
In propose work we are utilizing YOLOv8 CNN algorithm to detect various emergency vehicles such as Fire Engine, Ambulance and Police. We don’t have any sensors so we are reading random values which will be consider as temperature in Fahrenheit and using below conditions to identify weather.
If temperature < 5 then weather is FOG and signal duration = 35
If temperature > 5 and < 50 then weather is Rain and signal duration = 40
Else weather is normal then signal duration = 30
With above temperature if emergency vehicle detected then signal time automatically increased to 15 more seconds.
To implement this project we have designed following modules
1)	Upload Emergency Vehicle Dataset: using this module can upload dataset to application
2)	Pre-process Dataset: this module will read all images and then extract and normalize pixel values
3)	Train Emergency Vehicle Detection Algorithm: process features will be input to algorithm to train a model and this model will be applied on test data to calculate detection accuracy
4)	Training Graph: will plot training graph
5)	Emergency Vehicle Detection from Video: using this module user can upload test videos and then system will adjust signal time based on weather data and detected emergency vehicle
