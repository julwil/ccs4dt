Search.setIndex({docnames:["ccs4dt","ccs4dt.main","ccs4dt.main.http","ccs4dt.main.modules","ccs4dt.main.modules.clustering","ccs4dt.main.modules.conversion","ccs4dt.main.modules.correlation","ccs4dt.main.modules.data_management","ccs4dt.main.modules.prediction","ccs4dt.main.shared","ccs4dt.main.shared.database","ccs4dt.main.shared.enums","ccs4dt.tests","ccs4dt.tests.integration","ccs4dt.tests.unit","ccs4dt.tests.unit.modules","index","modules","scripts","scripts.synthetic_data_generation","setup","synthetic_data_generation","synthetic_data_generator","transform_test_data_set"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,sphinx:56},filenames:["ccs4dt.rst","ccs4dt.main.rst","ccs4dt.main.http.rst","ccs4dt.main.modules.rst","ccs4dt.main.modules.clustering.rst","ccs4dt.main.modules.conversion.rst","ccs4dt.main.modules.correlation.rst","ccs4dt.main.modules.data_management.rst","ccs4dt.main.modules.prediction.rst","ccs4dt.main.shared.rst","ccs4dt.main.shared.database.rst","ccs4dt.main.shared.enums.rst","ccs4dt.tests.rst","ccs4dt.tests.integration.rst","ccs4dt.tests.unit.rst","ccs4dt.tests.unit.modules.rst","index.rst","modules.rst","scripts.rst","scripts.synthetic_data_generation.rst","setup.rst","synthetic_data_generation.rst","synthetic_data_generator.rst","transform_test_data_set.rst"],objects:{"":{ccs4dt:[0,0,0,"-"],scripts:[18,0,0,"-"]},"ccs4dt.main":{http:[2,0,0,"-"],modules:[3,0,0,"-"],shared:[9,0,0,"-"]},"ccs4dt.main.http":{input_batch_controller:[2,0,0,"-"],location_controller:[2,0,0,"-"]},"ccs4dt.main.http.input_batch_controller":{get_all:[2,1,1,""],get_input_by_id:[2,1,1,""],get_output_by_id:[2,1,1,""],post:[2,1,1,""]},"ccs4dt.main.http.location_controller":{get_all:[2,1,1,""],get_by_id:[2,1,1,""],post:[2,1,1,""]},"ccs4dt.main.modules":{clustering:[4,0,0,"-"],conversion:[5,0,0,"-"],correlation:[6,0,0,"-"],data_management:[7,0,0,"-"],prediction:[8,0,0,"-"]},"ccs4dt.main.modules.conversion":{converter:[5,0,0,"-"]},"ccs4dt.main.modules.conversion.converter":{Converter:[5,2,1,""]},"ccs4dt.main.modules.conversion.converter.Converter":{add_sensor:[5,3,1,""],run:[5,3,1,""]},"ccs4dt.main.modules.data_management":{input_batch_service:[7,0,0,"-"],location_service:[7,0,0,"-"],process_batch_thread:[7,0,0,"-"]},"ccs4dt.main.modules.data_management.input_batch_service":{InputBatchService:[7,2,1,""]},"ccs4dt.main.modules.data_management.input_batch_service.InputBatchService":{create:[7,3,1,""],get_all:[7,3,1,""],get_by_id:[7,3,1,""],save_batch_to_influx:[7,3,1,""],update:[7,3,1,""],update_status:[7,3,1,""]},"ccs4dt.main.modules.data_management.location_service":{LocationService:[7,2,1,""]},"ccs4dt.main.modules.data_management.location_service.LocationService":{create:[7,3,1,""],get_all:[7,3,1,""],get_by_id:[7,3,1,""]},"ccs4dt.main.modules.data_management.process_batch_thread":{ProcessBatchThread:[7,2,1,""]},"ccs4dt.main.modules.data_management.process_batch_thread.ProcessBatchThread":{run:[7,3,1,""]},"ccs4dt.main.shared":{database:[10,0,0,"-"],enums:[11,0,0,"-"]},"ccs4dt.main.shared.database":{core_db:[10,0,0,"-"],influx_db:[10,0,0,"-"]},"ccs4dt.main.shared.database.core_db":{CoreDB:[10,2,1,""]},"ccs4dt.main.shared.database.core_db.CoreDB":{connection:[10,3,1,""]},"ccs4dt.main.shared.database.influx_db":{InfluxDB:[10,2,1,""]},"ccs4dt.main.shared.enums":{input_batch_status:[11,0,0,"-"],measurement_unit:[11,0,0,"-"]},"ccs4dt.main.shared.enums.input_batch_status":{InputBatchStatus:[11,2,1,""]},"ccs4dt.main.shared.enums.input_batch_status.InputBatchStatus":{FAILED:[11,4,1,""],FINISHED:[11,4,1,""],PROCESSING:[11,4,1,""],SCHEDULED:[11,4,1,""]},"ccs4dt.main.shared.enums.measurement_unit":{MeasurementUnit:[11,2,1,""]},"ccs4dt.main.shared.enums.measurement_unit.MeasurementUnit":{CENTIMETER:[11,4,1,""],METER:[11,4,1,""],MILLIMETER:[11,4,1,""]},"ccs4dt.tests":{integration:[13,0,0,"-"],unit:[14,0,0,"-"]},"ccs4dt.tests.integration":{test_input_batch_controller:[13,0,0,"-"],test_location_controller:[13,0,0,"-"]},"ccs4dt.tests.integration.test_input_batch_controller":{client:[13,1,1,""],get_input_batch_dummy:[13,1,1,""],test_get_all:[13,1,1,""],test_get_by_id:[13,1,1,""],test_get_outputs:[13,1,1,""],test_post:[13,1,1,""]},"ccs4dt.tests.integration.test_location_controller":{client:[13,1,1,""],get_location_dummy:[13,1,1,""],test_get_all:[13,1,1,""],test_get_by_id:[13,1,1,""],test_post:[13,1,1,""]},"ccs4dt.tests.unit":{modules:[15,0,0,"-"]},"ccs4dt.tests.unit.modules":{test_converter:[15,0,0,"-"]},"ccs4dt.tests.unit.modules.test_converter":{test_coordinate_offset_no_rotation:[15,1,1,""],test_coordinate_offset_with_rotation:[15,1,1,""]},"scripts.synthetic_data_generation":{synthetic_data_generator:[19,0,0,"-"]},"scripts.synthetic_data_generation.synthetic_data_generator":{Person:[19,2,1,""],generate_random_movement:[19,1,1,""]},"scripts.synthetic_data_generation.synthetic_data_generator.Person":{get_position:[19,3,1,""],get_position_x:[19,3,1,""],get_position_y:[19,3,1,""],get_straight_line:[19,3,1,""],set_position_x:[19,3,1,""],set_position_y:[19,3,1,""],walk_straight_line:[19,3,1,""]},ccs4dt:{main:[1,0,0,"-"],tests:[12,0,0,"-"]},scripts:{synthetic_data_generation:[19,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","function","Python function"],"2":["py","class","Python class"],"3":["py","method","Python method"],"4":["py","attribute","Python attribute"]},objtypes:{"0":"py:module","1":"py:function","2":"py:class","3":"py:method","4":"py:attribute"},terms:{"1":[5,19],"2":5,"21131fdc":19,"3":5,"4ed7":19,"6931d8c571c4":19,"972e":19,"class":[5,7,10,11,19],"enum":[1,9],"float":5,"int":7,"new":[2,7,10],"return":[2,7,10],The:5,add:5,add_sensor:5,aggreg:5,all:[2,7,10],an:[2,7,11],angl:5,ar:5,arg:7,associ:10,async:[2,7],axi:5,base:[5,7,10,11,19],batch:[2,5,7,10,11],batch_id:2,boundary_i:19,boundary_x:19,can:[2,5,11],centimet:11,client:13,cluster:[1,3],cm:[5,11],collect:5,common:5,compar:5,comput:2,configur:[5,10],connect:[7,10],content:17,convers:[1,3],convert:[1,3],coordin:5,core_db:[1,7,9],coredb:[7,10],correl:[1,3],counterclockwis:5,creat:[2,7],daemon:7,data:[7,10],data_manag:[1,3],databas:[1,7,9],deg:5,depend:5,deploi:5,dict:7,differ:5,e:5,end_position_i:19,end_position_x:19,ensur:5,f3ff:19,fail:11,finish:[2,11],flask:2,format:5,from:5,g:5,generate_random_mov:19,get:[2,7],get_al:[2,7],get_by_id:[2,7],get_input_batch_dummi:13,get_input_by_id:2,get_location_dummi:13,get_output_by_id:2,get_posit:19,get_position_i:19,get_position_x:19,get_straight_lin:19,given:2,group:7,handl:7,harmon:5,have:11,http:[0,1],id:[2,5,7],identifi:19,includ:5,index:16,influx_db:[1,7,9],influxdb:[7,10],input:[2,5,7,10,11],input_batch_control:[0,1],input_batch_id:7,input_batch_servic:[1,3],input_batch_statu:[1,9],inputbatchservic:7,inputbatchstatu:11,instal:5,integr:12,its:10,kwarg:7,list:[5,7],locat:[2,5,7,10],location_control:[0,1],location_id:[2,7],location_servic:[1,3],locationservic:7,m:[5,11],main:[0,17],manag:7,measur:[5,11],measurement_unit:[1,5,9],measurementunit:11,meter:11,millimet:11,mm:[5,11],modul:[16,17],name:7,new_statu:7,none:7,object:[5,7,10,19],object_speed_i:19,object_speed_x:19,offset:5,output:2,packag:17,page:16,paramet:[2,5,7],person:19,pitch:5,plane:5,poll:2,position_i:19,position_x:19,post:2,predict:[1,3],process:[2,7,11],process_batch_thread:[1,3],processbatchthread:7,provid:5,relat:5,respect:5,respons:[2,5,7],result:[2,7],roll:5,rotat:5,rtype:10,run:[5,7],sampling_r:19,save:7,save_batch_to_influx:7,schedul:11,script:17,search:16,sensor:[5,10,11],sensor_identifi:5,servic:7,set_position_i:19,set_position_x:19,setup:17,share:[0,1],space:5,speed:19,speed_i:19,speed_x:19,start:7,starting_posit:19,statu:[7,11],store:[7,10],str:[2,5,7,11],submodul:[0,1,3,9,12,14,17,18],subpackag:17,support:11,synthetic_data_gener:[17,18],system:5,target:7,test_convert:[12,14],test_coordinate_offset_no_rot:15,test_coordinate_offset_with_rot:15,test_get_al:13,test_get_by_id:13,test_get_output:13,test_input_batch_control:12,test_location_control:12,test_post:13,thei:5,thread:7,transform:5,transform_test_data_set:[17,18],type:[2,7],uniqu:5,unit:[5,11,12],updat:7,update_statu:7,valu:11,walk_straight_lin:19,when:2,where:5,within:5,x:5,x_origin:5,xy:5,xz:5,y:5,y_origin:5,yaw:5,yz:5,z:5,z_origin:5},titles:["ccs4dt package","ccs4dt.main package","ccs4dt.main.http package","ccs4dt.main.modules package","ccs4dt.main.modules.clustering package","ccs4dt.main.modules.conversion package","ccs4dt.main.modules.correlation package","ccs4dt.main.modules.data_management package","ccs4dt.main.modules.prediction package","ccs4dt.main.shared package","ccs4dt.main.shared.database package","ccs4dt.main.shared.enums package","ccs4dt.tests package","ccs4dt.tests.integration package","ccs4dt.tests.unit package","ccs4dt.tests.unit.modules package","Welcome to CCS4DT\u2019s documentation!","ccs4dt","scripts package","scripts.synthetic_data_generation package","setup module","synthetic_data_generation package","synthetic_data_generator module","transform_test_data_set module"],titleterms:{"enum":11,ccs4dt:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],cluster:4,content:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,18,19,21],convers:5,convert:5,core_db:10,correl:6,data_manag:7,databas:10,document:16,http:2,indic:16,influx_db:10,input_batch_control:2,input_batch_servic:7,input_batch_statu:11,integr:13,location_control:2,location_servic:7,main:[1,2,3,4,5,6,7,8,9,10,11],measurement_unit:11,modul:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,18,19,20,21,22,23],packag:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,18,19,21],predict:8,process_batch_thread:7,s:16,script:[18,19],setup:20,share:[9,10,11],submodul:[2,5,7,10,11,13,15,19,21],subpackag:[0,1,3,9,12,14,18],synthetic_data_gener:[19,21,22],tabl:16,test:[12,13,14,15],test_convert:15,test_input_batch_control:13,test_location_control:13,transform_test_data_set:[19,21,23],unit:[14,15],welcom:16}})