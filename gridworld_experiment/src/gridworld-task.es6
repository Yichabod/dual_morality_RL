/**
 * Created by markho on 6/25/17.
 */
import $ from 'jquery';
import forOwn from 'lodash/forown';
import map from 'lodash/map';
import fromPairs from 'lodash/frompairs';
import includes from 'lodash/includes';
import concat from 'lodash/concat';
const _ = {forOwn, map, fromPairs, includes, concat};
let GridWorldPainter = require("gridworld-painter");
import * as gwmdp from "./gridworld-mdp.es6";
var Raphael = require('raphael');
let GridWorldMDP = gwmdp.GridWorldMDP;

class GridWorldTask {
    constructor({
        container,
        step_callback = (d) => {console.log(d)},
        endtask_callback = () => {},

        OBJECT_ANIMATION_TIME = 200,
        REWARD_ANIMATION_TIME = 600,
        ITER_LIMIT = 5,
        disable_during_movement = true,
        disable_hold_key = true,
        WALL_WIDTH = .08,
        TILE_SIZE = 90,
        INTENTIONAL_ACTION_TIME_PROP = .4,
        DELAY_TO_REACTIVATE_UI = .8,
        END_OF_ROUND_DELAY_MULTIPLIER = 4,
        prevent_default_key_event = true
    }) {
        this.container = container;
        this.step_callback = step_callback;
        this.endtask_callback = endtask_callback;

        this.painter_config = {
            OBJECT_ANIMATION_TIME,
            WALL_WIDTH,
            TILE_SIZE
        };

        this.iter_limit = ITER_LIMIT;
        this.disable_during_movement = disable_during_movement;
        this.INTENTIONAL_ACTION_TIME_PROP = INTENTIONAL_ACTION_TIME_PROP;
        this.DELAY_TO_REACTIVATE_UI = DELAY_TO_REACTIVATE_UI;
        this.END_OF_ROUND_DELAY_MULTIPLIER = END_OF_ROUND_DELAY_MULTIPLIER;
        this.REWARD_ANIMATION_TIME = REWARD_ANIMATION_TIME;
        this.TILE_SIZE = TILE_SIZE;
        this.disable_hold_key = disable_hold_key;
        this.prevent_default_key_event = prevent_default_key_event;
        if (this.prevent_default_key_event) {
            this._disable_default_key_response()
        }
    }

    init({
        walls = [],
        init_state,
        switch_pos,
        targets,
        wait_time,
        time_limit,
        best_reward,

        feature_colors = {
            '.': 'white',
            'b': 'lightblue',
            'g': 'lightgreen'
        },
        show_rewards = true
        
    }) {
        let task_params = arguments[0];
        this.mdp = new GridWorldMDP(task_params);

        //initialize painter
        if (typeof(this.painter) === "undefined") {
            this.painter = new GridWorldPainter(
                this.mdp.width,
                this.mdp.height,
                this.container,
                this.painter_config
            );
            this.painter.initialize_paper();
        }

        let tile_params = _.fromPairs(_.map(this.mdp.state_features, (f, s) => {
            return [s, {fill: feature_colors[f]}]
        }));
        this.painter.draw_tiles(tile_params);

        this.painter.draw_walls(walls);

        // code to add all objects in init locations
        // text 1, text 2 and train are maintained outside of gridworld painter

        this.painter.add_object("circle", "switch1", {"fill" : "red",'r':25});
        this.painter.add_object("rect", "switch2", {"fill" : "black","object_length":.7, "object_width": .7});
        this.painter.draw_object(switch_pos[0],switch_pos[1], "<", "switch2");
        this.painter.draw_object(switch_pos[0],switch_pos[1], "<", "switch1");

        this.painter.add_object("circle", "agent", {"fill" : "black"});
        var agent_pos = init_state['agent']
        this.painter.draw_object(agent_pos[0], agent_pos[1], undefined, "agent");
        this.state = init_state;

        this.painter.add_object("rect","cargo2",{"fill":"lightblue","object_length":.7, "object_width": .7});
        var cargo2_pos = init_state['cargo2'];
        this.painter.draw_object(cargo2_pos[0], cargo2_pos[1], "<", "cargo2");
        this.text2 = this.painter.add_text(cargo2_pos[0], cargo2_pos[1], "2", {"font-size":40});

        this.painter.add_object("rect","cargo1",{"fill":"lightgreen","object_length":.7, "object_width": .7});
        var cargo1_pos = init_state['cargo1']
        this.painter.draw_object(cargo1_pos[0], cargo1_pos[1], "<", "cargo1");
        this.text1 = this.painter.add_text(cargo1_pos[0], cargo1_pos[1],"1", {"font-size":40});

        var train_pos = init_state['train'];
        this.vel_mapping = {"1,0":"1","-1,0":"2","0,1":"3","0,-1":"4"}
        var src = "/assets/train" + this.vel_mapping[String(init_state['trainvel'])] + ".png"
        
        if (init_state['trainvel'][0] == 0){
            this.train_width = 40
            this.train_height = 80
        } else {
            this.train_width = 80
            this.train_height = 40
        }
        
        var train_x = (train_pos[0] + .5)*this.painter.TILE_SIZE+this.painter.DISPLAY_BORDER-this.train_width/2
        var train_y = (this.painter.y_to_h(train_pos[1]) + .5)*this.painter.TILE_SIZE+this.painter.DISPLAY_BORDER-this.train_height/2
        this.train = this.painter.paper.image(src,train_x,train_y, this.train_width,this.train_height);
        this.train_transform = 0

        this.best_reward = best_reward
        this.show_rewards = show_rewards;

        //flags
        this.task_ended = false;
        this.task_paused = false;
        this.input_enabled = false;

        this.iter = 0;
        this.reward = 0;
        if (this.mdp.arraysEqual(targets['target1'],init_state['cargo1'])){
            this.reward += 1
        } 
        if (this.mdp.arraysEqual(targets['target2'],init_state['cargo2'])){
            this.reward += 1
        } 
        this.update_stats()

        this.wait_time = wait_time
        this.time_limit = time_limit
        this.time = 0 //to keep track of timed version

        this.data = {}
    }

    start() {
        if (this.wait_time>0){
            document.getElementById('timer').innerText = String(this.wait_time);
            this.time = this.wait_time-1
            var mytimer = window.setInterval(() => {
                document.getElementById('timer').innerText = String(this.time);
                this.time -= 1 
            }, 1000);
            setTimeout(() => {
                document.getElementById('timer').innerText = "BEGIN";
                this.start_datetime = new Date();
                this._enable_response();
                window.clearInterval(mytimer)
            }, this.wait_time*1000);

        } else if (this.time_limit != undefined) {
            document.getElementById('timer').innerText = String(this.time_limit);
            this.time = this.time_limit-1
            this.start_datetime = new Date();
            this._enable_response();
            var mytimer = window.setInterval(() => {
                document.getElementById('timer').innerText = String(this.time);
                this.time -= 1 
            }, 1000);
            setTimeout(() => {
                window.clearInterval(mytimer);
                document.getElementById('timer').innerText = "TIME OUT";
                this._end_task();
            }, this.time_limit*1000);
        }
        else {
            this.start_datetime = new Date();
            this._enable_response();
        }
    }

    end_task() {
        //Interface for client to set end task flag
        this.task_ended = true;
    }

    reset() {
        this._disable_response();
        this.painter.clear_objects();
        this.painter.draw_tiles();
    }

    clear() {
        this._disable_response();
        this._enable_default_key_response();
        this.painter.clear_objects();
        this.painter.draw_tiles();
    }

    move_agent(state) {
        this.state = state;
        this.painter.hide_object('agent');
        this.painter.draw_object(state[0], state[1], undefined, 'agent')
        this.painter.show_object('agent');
    }

    pause_next() {
        this.task_paused = true;
    }

    resume() {
        this.task_paused = false;
        this.start_datetime = +new Date;
        this._enable_response();
        // console.log("Task re-enabled time: "+(+new Date));
    }

    _disable_default_key_response() {
        $(document).on("keydown.disable_default", (e) => {
            let kc = e.keyCode ? e.keyCode : e.which;
            if ((kc === 37) || (kc === 38) || (kc === 39) || (kc === 40) || (kc === 32)) {
                e.preventDefault();
            }
        });
    }

    _enable_default_key_response() {
        $(document).off("keydown.disable_default");
    }

    _enable_response() {
        if (this.input_enabled) {
            return
        }
        this.input_enabled = true;
        $(document).on("keydown.task_response", (e) => {
            let kc = e.keyCode ? e.keyCode : e.which;
            let action;
            if (kc === 37) {
                action = "<";
            }
            else if (kc === 38) {
                action = "^";
            }
            else if (kc === 39) {
                action = ">";
            }
            else if (kc === 40) {
                action = "v";
            }
            else if (kc === 32) {
                action = "x";
            }
            else {
                return
            }
            this.last_key_code = kc;
            if (this.disable_during_movement) {
                this._disable_response();
            }
            
            this.data[this.iter] = [action,(new Date()-this.start_datetime)]
            this.start_datetime = new Date()
            let step_data = this._process_action({action});
            this.step_callback(step_data);
        });
    }

    _disable_response() {
        this.input_enabled = false;
        $(document).off("keydown.task_response");
    }

    _do_animation({reward, action, state, nextstate}) {
        var value = reward['value']
        let r_params = {
            fill: value < 0 ? 'red' : 'yellow',
            'stroke-width': 1.5,
            stroke: value < 0 ? 'white' : 'black',
            "font-size": this.TILE_SIZE/2
        };

        let r_string = value < 0 ? String(value) : "+" + value;

        this.painter.add_text(state['cargo1'][0],state['cargo1'][1],"")

        //animate agent
        this.painter.animate_object_movement({
            action: action,
            new_x: nextstate['agent'][0],
            new_y: nextstate['agent'][1],
            object_id: 'agent'
        });

        // animate cargo1
        this.painter.animate_object_movement({
            action: action,
            new_x: nextstate['cargo1'][0],
            new_y: nextstate['cargo1'][1],
            object_id: 'cargo1'
        });

        //animate cargo1 text individually - not included in painter lib
        var move1 = Raphael.animation({
            x : (nextstate['cargo1'][0] + .5)*this.painter.TILE_SIZE+this.painter.DISPLAY_BORDER,
            y : (this.painter.y_to_h(nextstate['cargo1'][1])+.5)*this.painter.TILE_SIZE+this.painter.DISPLAY_BORDER
        },this.painter.OBJECT_ANIMATION_TIME, 'easeInOut');
        this.text1.animate(move1);

        this.painter.animate_object_movement({
            action: action,
            new_x: nextstate['cargo2'][0],
            new_y: nextstate['cargo2'][1],
            object_id: 'cargo2'
        });

        var move2 = Raphael.animation({
            x : (nextstate['cargo2'][0] + .5)*this.painter.TILE_SIZE+this.painter.DISPLAY_BORDER,
            y : (this.painter.y_to_h(nextstate['cargo2'][1])+.5)*this.painter.TILE_SIZE+this.painter.DISPLAY_BORDER
        },this.painter.OBJECT_ANIMATION_TIME, 'easeInOut');
        this.text2.animate(move2);

        let ACTION_CODES = {
            '1,0' : '>',
            '-1,0' : '<',
            '0,-1' : 'v',
            '0,1' : '^',
            '0,0' : 'x'
        };

        var train_stopped = this.mdp.arraysEqual(state['trainvel'], [0,0])
        if (!this.mdp.arraysEqual(nextstate['trainvel'], state['trainvel']) && !train_stopped){
            if (state['trainvel'][0] != 0){
                this.train_transform -= 90
            } else {
                this.train_transform += 90
            }
            this.train_height, this.train_width = this.train_width, this.train_height
        }
        var transform = "r" + String(this.train_transform)
        if (state['trainvel'][0] != 0 || state['trainvel'][1] != 0){
            var move_train = Raphael.animation({
                x: (nextstate["train"][0] + .5)*this.painter.TILE_SIZE+this.painter.DISPLAY_BORDER-this.train_width/2,
                y: (this.painter.y_to_h(nextstate["train"][1]) + .5)*this.painter.TILE_SIZE+this.painter.DISPLAY_BORDER-this.train_height/2,
                transform: transform,
            }, this.painter.OBJECT_ANIMATION_TIME, 'easeInOut')
            this.train.animate(move_train)
        } 


        let animtime = this.painter.OBJECT_ANIMATION_TIME;
        // console.log("Animation-end time: "+((+new Date)+this.painter.OBJECT_ANIMATION_TIME));
        if (this.show_rewards && value !== 0) {
            setTimeout(() => {
                this.painter.float_text(
                    reward['position'][0],
                    reward['position'][1],
                    r_string,
                    r_params,
                    undefined,
                    undefined,
                    undefined,
                    undefined,
                    this.REWARD_ANIMATION_TIME
                );
            }, animtime)
        }
    }

    _end_task() {
        let animtime = this.painter.OBJECT_ANIMATION_TIME;
        
        var result_color = "green"
        if (this.reward<this.best_reward){ 
            result_color = "red"
        }
        this.painter.add_object("rect", "results", {"fill" : result_color,"object_length":2.5, "object_width":1.5});
        this.painter.draw_object(2,2, "<", "results");
        this.painter.add_object("rect", "results2", {"fill" : "white","object_length":2.25, "object_width":1.25});
        this.painter.draw_object(2,2, "<", "results2");
        var scoretext = "YOUR SCORE: " + String(this.reward) + "\nBEST SCORE: " + String(this.best_reward)
        scoretext += "\npress n to continue"
        this.painter.add_text(2,2, scoretext, {"font-size":20});

        this._disable_response();
        $(document).on("keydown.task_response", (e) => {
            let kc = e.keyCode ? e.keyCode : e.which;
            if (kc === 78) {
                this._disable_response();
                setTimeout(() => {
                    this.painter.paper.remove()
                    this.endtask_callback(this.data,this.reward);
                }, animtime*this.END_OF_ROUND_DELAY_MULTIPLIER);
            }
        })
    }

    _setup_trial() {
        let animtime = this.painter.OBJECT_ANIMATION_TIME;

        // Different conditions depending on pre-conditions
        // for next responses
        if (this.disable_during_movement) {
            if (this.disable_hold_key) {
                $(document).on("keyup.enable_resp", (e) => {
                    let kc = e.keyCode ? e.keyCode : e.which;
                    if (this.last_key_code !== kc) {
                        return
                    }
                    $(document).off("keyup.enable_resp");
                    this._key_unpressed = true;
                })
                setTimeout( () => {
                    if (!this._key_unpressed) {
                        $(document).off("keyup.enable_resp");
                        $(document).on("keyup.enable_resp", (e) => {
                            let kc = e.keyCode ? e.keyCode : e.which;
                            if (this.last_key_code !== kc) {
                                return
                            }
                            $(document).off("keyup.enable_resp");
                            this._enable_response();
                            this._key_unpressed = false;
                        });
                    }
                    else {
                        this._key_unpressed = false;
                        this._enable_response();
                    }

                    this.start_datetime = +new Date;
                }, animtime*this.DELAY_TO_REACTIVATE_UI);
            }
            else {
                setTimeout(() => {
                    this._enable_response();
                    this.start_datetime = +new Date;
                }, animtime*this.DELAY_TO_REACTIVATE_UI)
            }
        }
        else {
            console.warn("FEATURE NOT IMPLEMENTED!")
        }

    }

    update_stats(){
        var stats_text = "Reward = " + String(this.reward);
        stats_text += "\r\nStep = " + String(this.iter+1) + "/5"
        document.getElementById('reward').innerText = stats_text;
    }

    _process_action({action}) {
        let response_datetime = +new Date;
        let state, nextstate, reward;

        if (this.task_paused) {
            console.log("Response-disabled time: "(+new Date));
            this._disable_response();
        }
        else {
            state = this.state;
            nextstate = this.mdp.transition({state, action});
            reward = this.mdp.reward({state, action, nextstate});
            this.reward += reward['value'];
            this.data[this.iter].push(reward['value']);
            this.data[this.iter].push(this.reward);

            this._do_animation({reward, action, state, nextstate});

            if (this.mdp.is_terminal() || this.task_ended || this.iter >= this.iter_limit-1) {
                this.update_stats();
                this._end_task();
            }
            else {
                //This handles when/how to re-enable user responses
                this.update_stats();
                this._setup_trial();
            }
            this.state = nextstate;
        }

        this.iter += 1;

        return {
            state,
            state_type: this.mdp.state_features[state],
            action,
            nextstate,
            nextstate_type: this.mdp.state_features[nextstate],
            reward,
            start_datetime: this.start_datetime,
            response_datetime: response_datetime
        }
    }
}

if (typeof(window) === 'undefined') {
    module.exports = GridWorldTask;
}
else {
    window.GridWorldTask = GridWorldTask;
}

