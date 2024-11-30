The 'CityMeshAttributes' folder contains statistical data on feature elements of 1km*1km grids, with the feature data for each grid in the city Shanghai being stored in a dict format as a single json file, with the file name being the map ID of the grid. The original data is obtained from OpenStreetMap, and then further processed to obtain the feature vectors in the json files. The feature embeddings are spectral embeddings, trained through the SpectralNet. Here, we we directly provide the final trained results.
{
	"id": File name/Map ID,

    	"range": [Upper-left longitude, upper-left latitude, lower-right longitude, lower-right latitude],

    	"image_path": "CitySatellite\\Shanghai\\map ID.jpg",

    	"poi": [
        	[fine category label, quantity, major category label],
		......
		[fine category label, quantity, major category label]
    	],

    	"places": [
        	[fine category label, quantity, major category label],
		......
		[fine category label, quantity, major category label]
    	],

    	"transport": [
        	[fine category label, quantity, major category label],
		......
		[fine category label, quantity, major category label]
    	],

    	"traffic": [
        	[fine category label, quantity, major category label],
		......
		[fine category label, quantity, major category label]
    	],

    	"buildings": [the number of buildings, covered area],

    	"landuse": [
		[fine category label, quantity, major category label, covered area],
		......
		[fine category label, quantity, major category label, covered area]
    	],

    	"road_inner": [
        	[fine category label, quantity, major category label, length],
		......
		[fine category label, quantity, major category label, length]
    	],

    	"road_connect": {
        	"The map id of the mesh connected with by road": {
			"Road major category label": quantity,
			......
			"Road major category label": quantity
		}
        	......
		"The map id of the mesh connected with by road": {
			"Road major category label": quantity,
			......
			"Road major category label": quantity
		}

    	},

    	"population": [Number of population],

	"transport_embedding": [..........],

	"roadnet_embedding": [..........],
	
	"function_embedding": [..........],

	"city_embedding": [..........],

	"degree_embedding": [..........]
}
